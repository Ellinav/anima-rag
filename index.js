const { ProxyAgent } = require("undici");
const path = require("path");
const fs = require("fs");
const yaml = require("js-yaml");
const { LocalIndex } = require("vectra");
const AdmZip = require("adm-zip");
const bm25Engine = require("./bm25_engine");

let stProxyConfig = {
    enabled: false,
    url: "",
    bypass: ["localhost", "127.0.0.1"],
};

if (typeof global.File === "undefined") {
    console.log(
        "[Anima RAG] ⚠️ 检测到旧版 Node.js (< v20)，正在注入 File Polyfill...",
    );
    global.File = class File {
        constructor(fileBits, fileName, options) {
            this.name = fileName;
            this.type = options?.type || "";
            this.lastModified = options?.lastModified || Date.now();
            this._bits = fileBits;
        }
    };
}

// 🛠️ 兼容性补丁：防止 fetch 缺失 (Node < 18)
if (typeof global.fetch === "undefined") {
    console.error(
        "[Anima RAG] ❌ 致命错误: 当前 Node.js 版本过低，不支持 fetch。请升级至 Node 18+ (推荐 v20)",
    );
}

const VECTOR_ROOT = path.join(__dirname, "vectors");
const SESSION_ROOT = path.join(__dirname, "data", "sessions");
const activeIndexes = new Map();
const writeQueues = new Map();
const loadingPromises = new Map();
const SPECIAL_TAGS = [
    "Halloween",
    "Christmas",
    "Birthday",
    "Anniversary",
    "New Year",
    "Valentine",
    "Travel",
    "Period",
    "Sick",
];

let index;

// 🛠️ 配置：请填入你的 4096 维模型 API (DeepSeek/OpenAI等)
const EMBEDDING_CONFIG = {
    apiKey: "sk-xxxxxx",
    baseUrl: "https://api.openai.com/v1",
    model: "text-embedding-3-large",
};

// 🟢 [新增] 智能文本切片工具
function chunkText(text, strategy) {
    const { delimiter, chunkSize } = strategy;

    // 模式 A: 自定义分隔符 (优先)
    if (delimiter && delimiter.trim() !== "") {
        // 使用 split 分割，并过滤掉空行
        return text
            .split(delimiter)
            .map((t) => t.trim())
            .filter((t) => t.length > 0);
    }

    // 模式 B: 字符数 + 智能截断
    // 逻辑：每隔 chunkSize 切一刀，然后向后找最近的 \n 或 。
    const chunks = [];
    let startIndex = 0;
    const limit = parseInt(chunkSize) || 500;
    const totalLen = text.length;

    while (startIndex < totalLen) {
        let endIndex = startIndex + limit;

        if (endIndex >= totalLen) {
            endIndex = totalLen;
        } else {
            // 智能寻找断点：优先找换行，其次找句号/问号/感叹号
            // 在 limit 之后的 100 个字符内寻找，避免无限延长
            const searchWindow = text.substring(endIndex, endIndex + 100);

            // 1. 尝试找换行符
            let offset = searchWindow.indexOf("\n");

            // 2. 如果没换行，找句子结束符
            if (offset === -1) {
                const punctuationMatch = searchWindow.match(/[。.?!？！]/);
                if (punctuationMatch) {
                    offset = punctuationMatch.index;
                }
            }

            // 3. 如果找到了合适的断点，就延伸过去；否则硬切
            if (offset !== -1) {
                endIndex += offset + 1; // 包含标点
            }
        }

        const chunk = text.substring(startIndex, endIndex).trim();
        if (chunk) chunks.push(chunk);

        // 下一段从当前结束点开始
        startIndex = endIndex;
    }

    return chunks;
}

function loadSession(sessionId) {
    if (!sessionId) return { memories: [] };
    try {
        const safeId = sessionId.replace(
            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
            "_",
        );
        const filePath = path.join(SESSION_ROOT, `${safeId}.json`);

        if (!fs.existsSync(filePath)) return { memories: [] };

        const data = fs.readFileSync(filePath, "utf-8");
        return JSON.parse(data) || { memories: [] };
    } catch (e) {
        console.warn(
            `[Anima Session] Load failed for ${sessionId}: ${e.message}`,
        );
        return { memories: [] };
    }
}

// ✅ [新增] 保存会话记忆
function saveSession(sessionId, data) {
    if (!sessionId) return;
    try {
        const safeId = sessionId.replace(
            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
            "_",
        );

        // 确保目录存在
        if (!fs.existsSync(SESSION_ROOT)) {
            fs.mkdirSync(SESSION_ROOT, { recursive: true });
        }

        const filePath = path.join(SESSION_ROOT, `${safeId}.json`);
        fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf-8");
    } catch (e) {
        console.error(
            `[Anima Session] Save failed for ${sessionId}: ${e.message}`,
        );
    }
}

// ✅ [修改版] 核心回响逻辑 (含日志增强 + 存储瘦身 + 前端日志返回)
function processEchoLogic(
    currentResults,
    globalTop50,
    lastMemories,
    config = {},
) {
    const echoLogs = [];

    const maxTotalLimit = config.max_count ?? 10;
    const baseLife = config.base_life || 1;
    const impLife = config.imp_life || 2;
    const impTags = config.important_tags || ["important"];

    // 🔥 修改点 1：log 函数增加 meta 参数
    /**
     * @param {string} msg
     * @param {any} [meta] - 允许传入对象
     */
    const log = (msg, meta = null) => {
        console.log(msg);
        echoLogs.push({
            step: "Echo System",
            info: msg,
            meta: meta,
        });
    };

    // 头部日志没有 meta，保持原样
    log(
        `[Anima Echo] 🔍 开始回响判定 | 上轮记忆: ${Object.keys(lastMemories).length} | 本轮命中: ${currentResults.length} | 全局校验池: ${globalTop50.length}`,
    );

    const echoItems = [];
    const nextMemories = {};
    let remainingSlots = Math.max(0, maxTotalLimit - currentResults.length);
    const currentIds = new Set(currentResults.map((r) => r.item.id));
    const globalIds = new Set(globalTop50.map((r) => r.item.id));

    const freshScoreMap = new Map();
    currentResults.forEach((r) => freshScoreMap.set(r.item.id, r.score));
    globalTop50.forEach((r) => {
        if (!freshScoreMap.has(r.item.id))
            freshScoreMap.set(r.item.id, r.score);
    });

    // --- 1. 处理旧记忆 ---
    Object.values(lastMemories).forEach((memory) => {
        const memId = memory.item.id;
        const indexStr = memory.item.metadata?.index || "unknown";

        // 🟢 [核心修改]：优先从地图里拿最新分数，没有才用上一轮的旧分数
        const currentScore = freshScoreMap.has(memId)
            ? freshScoreMap.get(memId)
            : memory.score || 0;

        const metaData = {
            score: currentScore, // ✅ 使用最新分数展示
            tags: memory.item.metadata?.tags || [],
            index: indexStr,
        };

        // A. 自然命中 (Refreshed) - 满血复活
        if (currentIds.has(memId)) {
            const leanItem = { ...memory.item };
            delete leanItem.vector;
            nextMemories[memId] = {
                ...memory,
                life: memory.maxLife,
                item: leanItem,
                score: currentScore, // ✅ 同步更新保存的最新分数，防止存入旧数据
            };
            log(
                `[Anima Echo] ♻️ [刷新] [📦 ${memory.source}] Index ${indexStr} (自然命中) | Life: ${memory.maxLife}`,
                metaData,
            );
            return;
        }

        // B & C & D. 尝试回响
        if (globalIds.has(memId)) {
            // 只要 Life > 0 或者是刚刚被复活的 (Life 1)，就有资格尝试回响
            if (memory.life > 0) {
                if (remainingSlots > 0) {
                    // [C. 回响成功]
                    const leanItem = { ...memory.item };
                    delete leanItem.vector;
                    echoItems.push({
                        item: leanItem,
                        score: currentScore, // 🟢 [修改 1] 这里原本是 memory.score || 0，改为最新分数
                        _source_collection: memory.source || "memory",
                        _is_echo: true,
                    });
                    remainingSlots--;
                    const newLife = memory.life - 1;
                    nextMemories[memId] = {
                        ...memory,
                        life: newLife,
                        item: leanItem,
                        score: currentScore, // 🟢 [修改 2] 顺手确保下一次存入本地的也是最新分数
                    };
                    log(
                        `[Anima Echo] 🔗 [回响成功] [📦 ${memory.source}] Index ${indexStr} | 剩余Life: ${newLife}`,
                        metaData,
                    );
                } else {
                    // [D. 惜败 (排队)]
                    const newLife = memory.life - 1;
                    nextMemories[memId] = {
                        ...memory,
                        life: newLife,
                        item: { ...memory.item },
                        score: currentScore, // 🟢 [修改 3] 排队的记忆也要更新成最新分数
                    };

                    log(
                        `[Anima Echo] ⏳ [排队等待] [📦 ${memory.source}] Index ${indexStr} (无卡槽) | 剩余Life: ${newLife}`,
                        metaData,
                    );
                }
            } else {
                // Life 本来就是 0 (且没被复活)，那就真的死了
                log(
                    `[Anima Echo] 💀 [记忆枯竭] [📦 ${memory.source}] Index ${indexStr} (Life耗尽 -> 删除)`,
                    metaData,
                );
            }
        } else {
            // [B. 离题] - 直接移除，不进入 nextMemories
            log(
                `[Anima Echo] 💨 [遗忘] [📦 ${memory.source}] Index ${indexStr} (脱离相关性范围)`,
                metaData,
            );
        }
    });

    // --- 2. 注册新记忆 ---
    currentResults.forEach((res) => {
        const resId = res.item.id;
        const indexStr = res.item.metadata?.index || "unknown";

        if (!nextMemories[resId]) {
            const tags = res.item.metadata.tags || [];
            const isImportant = tags.some((t) =>
                impTags.includes(t.toLowerCase()),
            );
            const initialLife = isImportant ? impLife : baseLife;
            const leanItem = { ...res.item };
            delete leanItem.vector;

            nextMemories[resId] = {
                life: initialLife,
                maxLife: initialLife,
                item: leanItem,
                score: res.score,
                source: res._source_collection,
            };

            // 🔥 修改点 3：新记忆也要 meta
            log(
                `[Anima Echo] 🆕 [新增记忆] Index ${indexStr} | 初始Life: ${initialLife}`,
                {
                    score: res.score,
                    tags: tags,
                    index: indexStr,
                },
            );
        }
    });

    return { echoItems, nextMemories, echoLogs };
}

// 辅助：获取向量
async function getEmbedding(text, config) {
    if (!config || !config.key) throw new Error("API Key missing");
    try {
        const fetchUrl = `${config.url.replace(/\/+$/, "")}/embeddings`;

        // 🔥 [新增调试日志] 打印正在请求的 URL
        console.log(
            `[Anima Debug] Embedding Request -> URL: ${fetchUrl}, Model: ${config.model}`,
        );

        const response = await fetch(fetchUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${config.key}`,
            },
            body: JSON.stringify({
                input: text,
                model: config.model,
            }),
        });

        if (!response.ok) {
            const errText = await response.text();
            let cleanMessage = "Unknown API Error";
            try {
                // 尝试解析 JSON
                const errJson = JSON.parse(errText);
                // 优先取 error.message (OpenAI标准), 其次 message, 最后 raw
                cleanMessage =
                    errJson.error?.message ||
                    errJson.message ||
                    JSON.stringify(errJson);
            } catch (e) {
                // 解析失败，说明是 HTML (如 Nginx 报错页)
                // 1. 使用正则剥离所有标签
                let stripped = errText.replace(/<[^>]*>?/gm, "").trim();
                // 2. 截取前 100 字符，防止整页 HTML 文本刷屏
                // 3. 移除多余空白符
                cleanMessage = stripped.replace(/\s+/g, " ").substring(0, 100);
                if (!cleanMessage)
                    cleanMessage = `HTTP Error ${response.status}`;
            }
            throw new Error(cleanMessage);
        }

        const data = await response.json();
        return data.data[0].embedding;
    } catch (error) {
        // 🔥 [新增调试日志] 打印网络层面的错误原因
        console.error(
            "[Anima RAG] Embedding Failed (Network/Code):",
            error.cause || error,
        );
        throw error;
    }
}

// 🟢 [新增] 辅助：请求重排模型 (带超时控制)
async function fetchRerank(query, documents, config) {
    if (!config || !config.key || !config.url)
        throw new Error("Rerank API 配置缺失");
    if (!documents || documents.length === 0) return [];

    const controller = new AbortController();
    const timeoutSeconds = config.timeout || 30;
    const timeoutId = setTimeout(
        () => controller.abort(),
        timeoutSeconds * 1000,
    );

    try {
        console.log(
            `[Anima Rerank] 📡 发起重排请求 | 文档数: ${documents.length} | 模型: ${config.model}`,
        );
        const response = await fetch(config.url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${config.key}`,
            },
            body: JSON.stringify({
                model: config.model,
                query: query,
                documents: documents.map((d) => d.text), // 提取文本发送
            }),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP Error ${response.status}`);
        }

        const data = await response.json();
        // 确保返回的是按相关度从高到低排序的结果
        return data.results || [];
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === "AbortError") {
            throw new Error(`Rerank 请求超时 (${timeoutSeconds}s)`);
        }
        throw error;
    }
}

async function runInQueue(collectionId, task) {
    if (!writeQueues.has(collectionId)) {
        writeQueues.set(collectionId, Promise.resolve());
    }
    // 将任务追加到该 ID 的 Promise 链末尾
    const taskPromise = writeQueues.get(collectionId).then(() => task());
    writeQueues.set(
        collectionId,
        taskPromise.catch(() => {}),
    ); // 忽略错误防止阻塞队列
    return taskPromise;
}

// 🆕 新增：动态获取/创建 Index 实例的辅助函数
// ✨ [修改] 增加 allowCreate 参数，控制是否允许创建新库
async function getIndex(collectionId, allowCreate = true) {
    if (!collectionId) throw new Error("Collection ID is required");

    const safeName = collectionId.replace(
        /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
        "_",
    );

    if (activeIndexes.has(safeName)) return activeIndexes.get(safeName);
    if (loadingPromises.has(safeName)) return loadingPromises.get(safeName);

    const loadTask = (async () => {
        const collectionPath = path.join(VECTOR_ROOT, safeName);
        console.log(
            `[Anima Debug] 📂 Loading Index: ${safeName} (Create: ${allowCreate})`,
        );

        // ✨ [新增] 核心拦截逻辑：如果不允许创建，且文件夹不存在，直接返回 null
        if (!allowCreate && !fs.existsSync(collectionPath)) {
            console.log(`[Anima RAG] 🛑 查询跳过不存在的库: ${safeName}`);
            return null;
        }

        // 下面是原有的创建/加载逻辑
        if (!fs.existsSync(collectionPath))
            fs.mkdirSync(collectionPath, { recursive: true });

        const indexInstance = new LocalIndex(collectionPath);

        // 注意：isIndexCreated 会检查 index.json
        // 如果不允许创建，但在上面的 existsSync通过了（说明有文件夹但可能没json），
        // 这里 LocalIndex 可能会尝试创建 json。
        // 为了最严格的控制，可以在这里再加一层判断，但通常 vectra 的行为是安全的。
        if (!(await indexInstance.isIndexCreated())) {
            await indexInstance.createIndex({
                version: 1,
                metadata_config: { indexed: ["tags", "index", "batch_id"] },
            });
        }

        try {
            const stats = await indexInstance.listItems();
            console.log(
                `[Anima Debug] ✅ Index ${safeName} loaded with ${stats.length} items.`,
            );
        } catch (e) {}

        return indexInstance;
    })();

    loadingPromises.set(safeName, loadTask);
    try {
        const instance = await loadTask;

        // ✨ [新增] 如果 loadTask 返回 null (因为不允许创建)，这里也返回 null
        if (!instance) {
            return null;
        }

        instance["_debug_id"] = collectionId;
        activeIndexes.set(safeName, instance);
        return instance;
    } finally {
        // ✨ [新增] 只有在非 null 时才清理 promise，防止缓存了 null 状态
        // 不过为了简单，统一清理也没问题
        loadingPromises.delete(safeName);
    }
}

// 🕵️‍♂️ 调试增强版：安全查询
async function queryIndexSafe(indexInstance, vector, k, filter) {
    try {
        const safeFilter = filter || undefined;
        const arity = indexInstance.queryItems.length;

        // console.log(`[Anima Debug] 🔎 执行检索 | Arity: ${arity} | K: ${k} | Filter: ${safeFilter ? "有" : "无"}`);

        let results;

        // ⚡ 核心修复：只要参数个数 >= 4，都视为新版逻辑
        // 新版签名：queryItems(vector, queryString, topK, filter, minScore?)
        if (arity >= 4) {
            // 必须传第二个参数为 "" (空字符串) 来跳过文本匹配
            results = await indexInstance.queryItems(vector, "", k, safeFilter);
        }
        // 旧版逻辑 (v0.x)
        else {
            if (safeFilter) {
                results = await indexInstance.queryItems(vector, k, safeFilter);
            } else {
                results = await indexInstance.queryItems(vector, k);
            }
        }

        // console.log(`[Anima Debug] ✅ 检索返回 ${results ? results.length : 0} 条`);
        return results || [];
    } catch (e) {
        console.error(`[Anima CRITICAL] ❌ 检索函数崩溃:`, e);
        return [];
    }
}

async function queryMultiIndices(
    indices,
    vector,
    k,
    filter = null,
    taskTag = "RAG",
    recentWeight = 0,
    currentSessionId = null,
) {
    console.log(
        `[Anima Debug] [${taskTag}] 🚀 并行检索 ${indices.length} 个库...`,
    );

    const promises = indices.map(async (idx) => {
        // 原有的查询逻辑
        const results = await queryIndexSafe(idx, vector, k, filter);

        // 🔥 [修改这里] 为每个结果附带来源库的 ID
        return results.map((res) => {
            res._source_collection = idx._debug_id || "unknown_lib";

            // 🟢 [新增核心逻辑] 近因加权：只对当前数据库的分数进行提升
            if (
                recentWeight > 0 &&
                currentSessionId &&
                res._source_collection === currentSessionId
            ) {
                const oldScore = res.score; // 记录原始分数
                res.score += Number(recentWeight);
                res._is_weighted = true; // 打上标记，供前端日志读取

                // 🟢 [优化] 打印后端加权日志，带上步骤和来源数据库
                const indexStr = res.item.metadata?.index || "unknown";
                console.log(
                    `[Anima 加权] [${taskTag}] 📦 ${res._source_collection} | ⬆️ Index ${indexStr} | 分数: ${oldScore.toFixed(4)} -> ${res.score.toFixed(4)} (+${recentWeight})`,
                );
            }

            return res;
        });
    });

    const resultsArrays = await Promise.all(promises);

    // 拍平结果
    let allResults = resultsArrays.flat();
    console.log(
        `[Anima Debug] [${taskTag}] 📊 聚合所有库结果，共 ${allResults.length} 条 (排序前)`,
    );

    // 排序
    allResults.sort((a, b) => b.score - a.score);

    // 截取
    return allResults.slice(0, 50);
}

// 🔥 修复版：动态策略执行器 (100% 保留原生后续步骤逻辑，采用拦截器模式重排)
async function performDynamicStrategy(indices, vector, config, ignoreIds = []) {
    let finalResults = [];
    let usedIds = new Set();
    let debugLogs = [];
    let echoCandidatePool = [];
    const steps = config.steps || [];
    const multiplier = config.global_multiplier || 2;
    const globalMinScore = config.min_score || 0;
    const recentWeight = config.recent_weight || 0;
    const currentSessionId = config.current_session_id || null;

    // 🟢 提取我们附加在 config 上的重排参数
    const searchText = config.searchText;
    const rerankConfig = config.rerankConfig || {};

    console.log(
        `[Anima RAG] 🚀 执行策略 | 步骤数: ${steps.length} | 排除ID: ${ignoreIds.length} 个 [${ignoreIds.join(", ")}]`,
    );

    // =========================================================
    // 原生：动态构建“功能性标签池”
    // =========================================================
    let functionalTagsSet = new Set();
    steps.forEach((s) => {
        if (["important", "status", "period", "special"].includes(s.type)) {
            if (s.labels && Array.isArray(s.labels)) {
                s.labels.forEach((l) => functionalTagsSet.add(l.toLowerCase()));
            }
            if (s.target_tag) {
                functionalTagsSet.add(s.target_tag.toLowerCase());
            }
        }
    });

    const buildFilter = (stepFilter = {}) => {
        if (!ignoreIds || ignoreIds.length === 0) return stepFilter;
        return { ...stepFilter, index: { $nin: ignoreIds } };
    };

    let detectedVibeTag = null;
    let detectedImportantLabels = [];

    // =========================================================
    // 🧠 新增：重排专用缓存池
    // =========================================================
    let basePool = [];
    let importantPool = [];
    let baseStepConfig = null;
    let importantStepConfig = null;

    // 🧠 新增：重排结算函数 (在进入 Status/Diversity 前执行，确保 usedIds 同步)
    const executeRerankFlush = async () => {
        if (!baseStepConfig && !importantStepConfig) return; // 没有需要重排的数据

        // 降级与兜底逻辑 (如果没开重排或重排失败，按原生分数录入)
        const fallback = () => {
            if (baseStepConfig) {
                let added = 0;
                for (const res of basePool) {
                    if (added >= baseStepConfig.count) break;
                    if (usedIds.has(res.item.id)) continue;
                    if (res.score < globalMinScore) continue;
                    finalResults.push(res);
                    usedIds.add(res.item.id);
                    added++;
                    debugLogs.push({
                        step: "Step 1: BASE",
                        library: res._source_collection,
                        uniqueID: res.item.metadata.index,
                        tags: (res.item.metadata.tags || []).join(", "),
                        score: res._is_weighted
                            ? `${res.score.toFixed(4)} (⬆️+${recentWeight})`
                            : res.score.toFixed(4),
                    });
                }
            }
            if (importantStepConfig && importantStepConfig.labels) {
                const stepThreshold = Math.max(0, globalMinScore - 0.2);
                importantStepConfig.labels.forEach((label) => {
                    let countForThisLabel = 0;
                    for (const res of importantPool) {
                        if (countForThisLabel >= importantStepConfig.count)
                            break;
                        if (usedIds.has(res.item.id)) continue;

                        // 🔒 仅在未重排时，Important 遵守降分门槛
                        if (res.score < stepThreshold) continue;

                        const tags = (res.item.metadata.tags || []).map((t) =>
                            t.toLowerCase(),
                        );
                        if (tags.includes(label.toLowerCase())) {
                            finalResults.push(res);
                            usedIds.add(res.item.id);
                            countForThisLabel++;
                            debugLogs.push({
                                step: "Step 2: IMPORTANT",
                                library: res._source_collection,
                                uniqueID: res.item.metadata.index,
                                tags: (res.item.metadata.tags || []).join(", "),
                                score: res._is_weighted
                                    ? `${res.score.toFixed(4)} (⬆️+${recentWeight})`
                                    : res.score.toFixed(4),
                            });
                        }
                    }
                });
            }
        };

        if (rerankConfig.enabled && rerankConfig.api && searchText) {
            try {
                const rCount = parseInt(rerankConfig.count) || 30;
                const halfCount = Math.floor(rCount / 2);

                const toRerankBase = basePool.slice(0, halfCount);
                const toRerankImp = importantPool.slice(
                    0,
                    rCount - toRerankBase.length,
                );

                const rerankMap = new Map();
                [...toRerankBase, ...toRerankImp].forEach((item) => {
                    if (!rerankMap.has(item.item.id)) {
                        rerankMap.set(item.item.id, {
                            id: item.item.id,
                            text: item.item.metadata.text,
                            originalData: item,
                        });
                    }
                });

                const documentsToSend = Array.from(rerankMap.values());
                const rerankResults = await fetchRerank(
                    searchText,
                    documentsToSend,
                    rerankConfig.api,
                );

                if (rerankResults && rerankResults.length > 0) {
                    console.log(`[Anima Rerank] 🎯 重排成功！处理分配...`);
                    const rankedItems = rerankResults.map((r) => {
                        const originalObj =
                            documentsToSend[r.index].originalData;
                        originalObj._rerank_score = r.relevance_score;
                        return originalObj;
                    });

                    // 1. 分配给 Base
                    if (baseStepConfig) {
                        let added = 0;
                        for (const res of rankedItems) {
                            if (added >= baseStepConfig.count) break;
                            if (usedIds.has(res.item.id)) continue;
                            finalResults.push(res);
                            usedIds.add(res.item.id);
                            added++;
                            debugLogs.push({
                                step: "Rerank: BASE",
                                library: res._source_collection,
                                uniqueID: res.item.metadata.index,
                                tags: (res.item.metadata.tags || []).join(", "),
                                score: `${res._is_weighted ? `${res.score.toFixed(4)} (⬆️+${recentWeight})` : res.score.toFixed(4)} ➡️ 精排: ${res._rerank_score.toFixed(4)}`,
                            });
                        }
                    }

                    // 2. 分配给 Important (按标签)
                    if (importantStepConfig && importantStepConfig.labels) {
                        importantStepConfig.labels.forEach((label) => {
                            let countForThisLabel = 0;
                            for (const res of rankedItems) {
                                if (
                                    countForThisLabel >=
                                    importantStepConfig.count
                                )
                                    break;
                                if (usedIds.has(res.item.id)) continue;

                                const tags = (res.item.metadata.tags || []).map(
                                    (t) => t.toLowerCase(),
                                );
                                if (tags.includes(label.toLowerCase())) {
                                    finalResults.push(res);
                                    usedIds.add(res.item.id);
                                    countForThisLabel++;
                                    debugLogs.push({
                                        step: "Rerank: IMPORTANT",
                                        library: res._source_collection,
                                        uniqueID: res.item.metadata.index,
                                        tags: (
                                            res.item.metadata.tags || []
                                        ).join(", "),
                                        score: `精排: ${res._rerank_score.toFixed(4)}`,
                                    });
                                }
                            }
                        });
                    }
                } else {
                    fallback();
                }
            } catch (e) {
                console.error(
                    `[Anima Rerank] ❌ 重排失败，回退粗排逻辑:`,
                    e.message,
                );
                fallback();
            }
        } else {
            fallback();
        }

        // 结算完毕，清空拦截缓存
        baseStepConfig = null;
        importantStepConfig = null;
    };

    // =========================================================
    // 循环执行步骤
    // =========================================================
    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        if (step.count <= 0) continue;

        // 🟢 核心：当遇到非 Base 和 Important 时，说明粗筛完毕，立即结算重排，
        // 这样后续的 Status/Diversity 才能正确识别 usedIds，避免重复并正确截取 Top N
        if (step.type !== "base" && step.type !== "important") {
            await executeRerankFlush();
        }

        let stepCoeff = 1.0;
        let useGlobalMultiplier = true;

        switch (step.type) {
            case "base":
                stepCoeff = 5;
                break;
            case "important":
                stepCoeff = 2;
                break;
            case "diversity":
                stepCoeff = 1.5;
                break;
            default:
                stepCoeff = 1.5;
                break;
        }

        const finalMultiplier = useGlobalMultiplier
            ? multiplier * stepCoeff
            : stepCoeff;
        let candidateK = Math.max(Math.ceil(step.count * finalMultiplier), 2);
        if (
            rerankConfig.enabled &&
            (step.type === "base" || step.type === "important")
        ) {
            const rCount = parseInt(rerankConfig.count) || 30;
            const halfQuota = Math.ceil(rCount / 2);
            candidateK = Math.max(candidateK, halfQuota);
        }
        console.log(
            `[Step ${i + 1} - ${step.type}] Count: ${step.count} | Multiplier: ${finalMultiplier.toFixed(1)}x | Candidates Per DB: ${candidateK}`,
        );

        let candidates = [];

        switch (step.type) {
            case "base":
                baseStepConfig = step; // 拦截交给重排
                candidates = await queryMultiIndices(
                    indices,
                    vector,
                    candidateK,
                    buildFilter({}),
                    "Step: BASE",
                    recentWeight,
                    currentSessionId,
                );

                // 🌟 原生 Vibe Tag 捕获逻辑 (完全没改)
                if (!detectedVibeTag && candidates.length > 0) {
                    const topItem = candidates[0].item.metadata;
                    const topTags = topItem.tags || [];
                    detectedVibeTag = topTags.find(
                        (t) => !functionalTagsSet.has(t.toLowerCase()),
                    );
                    if (detectedVibeTag) {
                        console.log(
                            `   [Base] 捕获 Vibe: ${detectedVibeTag} (已排除功能词)`,
                        );
                    }
                }
                candidates.sort((a, b) => b.score - a.score);

                if (echoCandidatePool.length === 0) {
                    echoCandidatePool = candidates.slice(0, 50);
                    console.log(
                        `[Anima Strategy] 🌊 已捕获 Base 全局池用于回响 (Top ${echoCandidatePool.length})`,
                    );
                }
                basePool = candidates.slice(0, 50); // 存入重排池
                continue; // 🚨 跳过原生聚合循环

            case "important":
                importantStepConfig = step; // 拦截交给重排
                if (step.labels && step.labels.length > 0) {
                    detectedImportantLabels = step.labels; // 记录用于后续排除
                    const impPromises = step.labels.map((label) =>
                        queryMultiIndices(
                            indices,
                            vector,
                            candidateK,
                            buildFilter({ tags: { $in: [label] } }),
                            "Step: IMPORTANT",
                            recentWeight,
                            currentSessionId,
                        ),
                    );
                    const impResults = await Promise.all(impPromises);

                    // 👇 完全恢复你的分路均衡逻辑，确保每个标签都能公平地进入重排池
                    let tempCandidates = [];
                    const tempUsedIds = new Set(usedIds);
                    const isRerankActive =
                        rerankConfig.enabled && rerankConfig.api && searchText;
                    const stepThreshold = isRerankActive
                        ? -999
                        : Math.max(0, globalMinScore - 0.2);

                    impResults.forEach((list) => {
                        list.sort((a, b) => b.score - a.score);
                        let countForThisLabel = 0;

                        const poolLimitForLabel = Math.max(step.count * 3, 5);

                        for (const res of list) {
                            if (countForThisLabel >= poolLimitForLabel) break;
                            if (tempUsedIds.has(res.item.id)) continue;

                            // 这里会自动根据 isRerankActive 决定是否卡分数
                            if (res.score < stepThreshold) continue;

                            tempCandidates.push(res);
                            tempUsedIds.add(res.item.id);
                            countForThisLabel++;
                        }
                    });

                    importantPool = tempCandidates; // 将均衡提取的候选人放入重排池
                    console.log(
                        `   [Important] 分路检索放入重排池: 触发 ${step.labels.join(", ")}`,
                    );
                }
                continue; // 🚨 跳过原生聚合循环，去下一步

            case "status":
                if (step.labels && step.labels.length > 0) {
                    const statusPromises = step.labels.map((label) =>
                        queryMultiIndices(
                            indices,
                            vector,
                            candidateK,
                            buildFilter({ tags: { $in: [label] } }),
                            "Step: STATUS",
                            recentWeight,
                            currentSessionId,
                        ),
                    );
                    const statusResults = await Promise.all(statusPromises);

                    candidates = [];
                    const tempUsedIds = new Set(usedIds);
                    const stepThreshold = Math.max(0, globalMinScore - 0.2);

                    statusResults.forEach((list) => {
                        list.sort((a, b) => b.score - a.score);
                        let countForThisLabel = 0;
                        for (const res of list) {
                            if (countForThisLabel >= step.count) break;
                            if (tempUsedIds.has(res.item.id)) continue;
                            candidates.push(res);
                            tempUsedIds.add(res.item.id);
                            countForThisLabel++;
                        }
                    });
                }
                break;

            case "period":
                if (step.labels && step.labels.length > 0) {
                    const periodPromises = step.labels.map((label) =>
                        queryMultiIndices(
                            indices,
                            vector,
                            candidateK,
                            buildFilter({ tags: { $in: [label] } }),
                            "Step: PERIOD",
                            recentWeight,
                            currentSessionId,
                        ),
                    );
                    const periodResults = await Promise.all(periodPromises);

                    candidates = [];
                    const tempUsedIds = new Set(usedIds);
                    const stepThreshold = Math.max(0, globalMinScore - 0.2);

                    periodResults.forEach((list) => {
                        list.sort((a, b) => b.score - a.score);
                        let countForThisLabel = 0;
                        for (const res of list) {
                            if (countForThisLabel >= step.count) break;
                            if (tempUsedIds.has(res.item.id)) continue;
                            candidates.push(res);
                            tempUsedIds.add(res.item.id);
                            countForThisLabel++;
                        }
                    });
                    console.log(
                        `   [Period] 生理分支检索: 触发 ${step.labels.join(", ")}`,
                    );
                }
                break;

            case "special":
                if (step.labels && step.labels.length > 0) {
                    const specialPromises = step.labels.map((label) =>
                        queryMultiIndices(
                            indices,
                            vector,
                            candidateK,
                            buildFilter({ tags: { $in: [label] } }),
                            "Step: SPECIAL",
                            recentWeight,
                            currentSessionId,
                        ),
                    );
                    const specialResults = await Promise.all(specialPromises);

                    candidates = [];
                    const tempUsedIds = new Set(usedIds);
                    const stepThreshold = Math.max(0, globalMinScore - 0.2);

                    specialResults.forEach((list) => {
                        list.sort((a, b) => b.score - a.score);
                        let countForThisLabel = 0;
                        for (const res of list) {
                            if (countForThisLabel >= step.count) break;
                            if (tempUsedIds.has(res.item.id)) continue;
                            candidates.push(res);
                            tempUsedIds.add(res.item.id);
                            countForThisLabel++;
                        }
                    });
                } else if (step.target_tag) {
                    candidates = await queryMultiIndices(
                        indices,
                        vector,
                        candidateK,
                        buildFilter({ tags: { $in: [step.target_tag] } }),
                        "Chat",
                        recentWeight,
                        currentSessionId,
                    );
                }
                break;

            case "diversity":
                // 🌟 原生丰富度逻辑 (完全没改)
                const excludeTags = [...detectedImportantLabels];
                if (detectedVibeTag && !excludeTags.includes(detectedVibeTag)) {
                    excludeTags.push(detectedVibeTag);
                }

                if (excludeTags.length > 0) {
                    candidates = await queryMultiIndices(
                        indices,
                        vector,
                        candidateK,
                        buildFilter({ tags: { $nin: excludeTags } }),
                        "Step: DIVERSITY",
                        recentWeight,
                        currentSessionId,
                    );
                } else {
                    candidates = await queryMultiIndices(
                        indices,
                        vector,
                        candidateK,
                        buildFilter({}),
                        "Step: DIVERSITY",
                        recentWeight,
                        currentSessionId,
                    );
                }
                break;
        }

        // === 原生聚合结果 (仅限 Status/Period/Special/Diversity) ===
        let addedInStep = 0;
        candidates.sort((a, b) => b.score - a.score);
        const limit =
            ["status", "important", "special", "period"].includes(step.type) &&
            step.labels
                ? step.count * step.labels.length
                : step.count;

        for (const res of candidates) {
            if (addedInStep >= limit) break;
            if (usedIds.has(res.item.id)) continue;
            finalResults.push(res);
            usedIds.add(res.item.id);
            addedInStep++;
            debugLogs.push({
                step: `Step ${i + 1}: ${step.type.toUpperCase()}`,
                library: res._source_collection,
                uniqueID: res.item.metadata.index,
                tags: (res.item.metadata.tags || []).join(", "),
                score: res._is_weighted
                    ? `${res.score.toFixed(4)} (⬆️+${recentWeight})`
                    : res.score.toFixed(4),
            });
        }
    }

    // 防御性调用：万一 base/important 是最后一步
    await executeRerankFlush();

    // =========================================================
    // 原生：时间序列最终排序
    // =========================================================
    finalResults.sort((a, b) => {
        const itemA = a.item.metadata;
        const itemB = b.item.metadata;

        const timeA = new Date(itemA.timestamp || 0).getTime();
        const timeB = new Date(itemB.timestamp || 0).getTime();
        if (timeA > 0 && timeB > 0 && timeA !== timeB) {
            return timeA - timeB;
        }

        const parseId = (str) => {
            const parts = (str || "0_0").split("_");
            return {
                batch: parseInt(parts[0] || 0),
                slice: parseInt(parts[1] || 0),
            };
        };

        const idA = parseId(itemA.index);
        const idB = parseId(itemB.index);

        if (idA.batch !== idB.batch) {
            return idA.batch - idB.batch;
        }
        return idA.slice - idB.slice;
    });

    finalResults["_debug_logs"] = debugLogs;
    finalResults["_echo_pool"] = echoCandidatePool;

    return finalResults;
}

async function init(router) {
    if (!fs.existsSync(VECTOR_ROOT)) {
        fs.mkdirSync(VECTOR_ROOT, { recursive: true });
    }
    console.log("[Anima RAG] 向量存储根目录就绪:", VECTOR_ROOT);

    // ✨ 新增：自动读取 SillyTavern 全局的 config.yaml 代理配置
    try {
        const configPath = path.join(__dirname, "../../config.yaml");
        if (fs.existsSync(configPath)) {
            const configData = yaml.load(fs.readFileSync(configPath, "utf8"));
            if (
                configData &&
                configData.requestProxy &&
                configData.requestProxy.enabled
            ) {
                stProxyConfig.enabled = true;
                stProxyConfig.url = configData.requestProxy.url;
                if (Array.isArray(configData.requestProxy.bypass)) {
                    stProxyConfig.bypass = configData.requestProxy.bypass;
                }
                console.log(
                    `[Anima Proxy] 🛡️ 已继承 ST 全局代理配置: ${stProxyConfig.url}`,
                );
            } else {
                console.log(
                    `[Anima Proxy] ⚡ ST 未开启代理，API 请求将使用本机直连网络`,
                );
            }
        }
    } catch (e) {
        console.warn(
            "[Anima Proxy] ⚠️ 读取 ST config.yaml 失败，将默认使用直连:",
            e.message,
        );
    }

    // API: 存入
    router.post("/insert", async (req, res) => {
        // 1. 解构请求数据
        const {
            collectionId,
            text,
            tags,
            timestamp,
            apiConfig,
            index,
            batch_id,
            bm25Config,
        } = req.body;

        if (
            !text ||
            typeof text !== "string" ||
            text.trim().length === 0 ||
            text === "(条目已丢失)" // 🟢 拦截特定错误文本
        ) {
            console.warn(
                `[Anima RAG] ⚠️ 拒绝写入无效文本 (Index: ${index}) Content: ${text}`,
            );
            return res.status(400).json({
                success: false,
                message: "Text content is invalid or missing.",
            });
        }

        // 🛡️ 安全处理 batch_id (这是修复的核心)
        // 如果前端传来的 batch_id 是 undefined 或 null，parseInt 会变成 NaN
        // 我们这里做一个判断：如果是 NaN，就强制设为 -1 或 0
        let safeBatchId = parseInt(batch_id);
        if (isNaN(safeBatchId)) {
            safeBatchId = -1;
        }

        let safeTimestamp = Number(timestamp);
        if (isNaN(safeTimestamp) || safeTimestamp <= 0) {
            // 如果传来的是 ISO 字符串，转为数字
            if (typeof timestamp === "string") {
                safeTimestamp = new Date(timestamp).getTime();
            }
            // 如果还是无效（比如 null/undefined），使用当前时间兜底
            if (isNaN(safeTimestamp) || safeTimestamp <= 0) {
                safeTimestamp = Date.now();
            }
        }

        try {
            await runInQueue(collectionId, async () => {
                const vector = await getEmbedding(text, apiConfig);
                const targetIndex = await getIndex(collectionId);

                // =========================================================
                // 🧹 步骤 0: 写入前自检，清理旧的同名 Index (防重复核心)
                // =========================================================
                if (index !== undefined && index !== null) {
                    const allItems = await targetIndex.listItems();

                    // 1. 找出旧的同名切片
                    const duplicates = allItems.filter(
                        (item) =>
                            item.metadata &&
                            String(item.metadata.index) === String(index),
                    );

                    if (duplicates.length > 0) {
                        console.log(
                            `[Anima RAG] 🔄 更新检测: 发现 Index ${index} 的旧版本 ${duplicates.length} 个，正在覆盖...`,
                        );

                        // 2. 构建删除计划
                        const safeName = collectionId.replace(
                            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                            "_",
                        );
                        const collectionPath = path.join(VECTOR_ROOT, safeName);

                        const deletionPlan = duplicates.map((item) => ({
                            id: item.id,
                            filePath: item.metadataFile
                                ? path.join(collectionPath, item.metadataFile)
                                : null,
                        }));

                        // 3. 执行物理 + 逻辑删除
                        const idsToDeleteFromBm25 = [];
                        for (const plan of deletionPlan) {
                            try {
                                await targetIndex.deleteItem(plan.id); // 删向量索引
                                idsToDeleteFromBm25.push(plan.id); // 🟢 记录旧 ID

                                if (
                                    plan.filePath &&
                                    fs.existsSync(plan.filePath)
                                ) {
                                    fs.unlinkSync(plan.filePath); // 删文件
                                }
                            } catch (e) {
                                console.warn(
                                    `[Anima] 覆盖清理旧文件失败: ${e.message}`,
                                );
                            }
                        }

                        // 🟢 新增：把旧数据的幽灵从 BM25 引擎中彻底抹除
                        if (
                            idsToDeleteFromBm25.length > 0 &&
                            bm25Config &&
                            bm25Config.enabled
                        ) {
                            try {
                                await bm25Engine.deleteDocuments(
                                    collectionId,
                                    idsToDeleteFromBm25,
                                );
                            } catch (bm25DelErr) {
                                console.error(
                                    `[Anima BM25] 清理旧幽灵数据失败:`,
                                    bm25DelErr,
                                );
                            }
                        }
                    }
                }

                // =========================================================
                // 📝 步骤 1: 插入新版本
                // =========================================================
                const newItem = await targetIndex.insertItem({
                    vector: vector,
                    metadata: {
                        text,
                        tags,
                        timestamp: safeTimestamp,
                        index,
                        batch_id: safeBatchId,
                    },
                });

                console.log(
                    `[Anima RAG] ✅ 写入成功 | Batch: ${safeBatchId} | Index: ${index}`,
                );
                if (bm25Config && bm25Config.enabled) {
                    try {
                        const dict = bm25Config.dictionary || [];

                        await bm25Engine.upsertDocument(
                            collectionId,
                            {
                                id: newItem.id, // 使用和向量库相同的 ID，方便以后对照
                                text,
                                tags,
                                timestamp: safeTimestamp,
                                index,
                                batch_id: safeBatchId,
                            },
                            dict,
                            "chat",
                        );
                    } catch (bm25Err) {
                        console.error(`[Anima BM25] ❌ 同步写入失败:`, bm25Err);
                        // 注意：BM25 写入失败不应该阻塞响应，仅打印日志
                    }
                }

                res.json({ success: true, vectorId: newItem.id });
            });
        } catch (err) {
            console.error("[Anima RAG Insert Error]", err);
            // 防止 headers 已经发送的情况
            if (!res.headersSent) {
                // 修改：改为返回 JSON 对象，不要直接传 err.message
                res.status(500).json({
                    success: false,
                    message: err.message || "未知后端错误",
                });
            }
        }
    });

    router.post("/test_connection", async (req, res) => {
        const { apiConfig } = req.body;

        if (!apiConfig || !apiConfig.key) {
            return res.status(400).send("缺少 API 配置或 Key");
        }

        try {
            console.log(
                `[Anima RAG] 🧪 正在测试连接: ${apiConfig.model} @ ${apiConfig.url}`,
            );

            // 使用 "Hello World" 进行一次极简的向量化测试
            const vector = await getEmbedding("Test Connection", apiConfig);

            if (vector && vector.length > 0) {
                res.json({
                    success: true,
                    message: `连接成功！向量维度: ${vector.length}`,
                    dimension: vector.length,
                });
            } else {
                throw new Error("API 返回了空向量");
            }
        } catch (err) {
            console.error(`[Anima RAG] 测试失败: ${err.message}`);
            // 将错误信息返回给前端
            res.status(500).send(err.message);
        }
    });

    router.post("/import_knowledge", async (req, res) => {
        // 🌟 1. 接收 vectorConfig
        const {
            fileName,
            fileContent,
            settings,
            apiConfig,
            bm25Config,
            vectorConfig,
        } = req.body;

        if (!fileName || !fileContent)
            return res.status(400).send("Missing file data");

        // 🌟 2. 判断是否写入向量库
        const writeVector = vectorConfig ? vectorConfig.enabled : true;

        const safeName = fileName
            .replace(/\.[^/.]+$/, "")
            .replace(/[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g, "_");
        const collectionId = `kb_${safeName}`;

        console.log(
            `[Anima KB] 📚 处理知识库: ${collectionId} | Vector: ${writeVector} | BM25: ${bm25Config?.enabled}`,
        );

        try {
            await runInQueue(collectionId, async () => {
                let cleanIndex = null;

                // 🌟 3. 如果开启了向量，才去清空旧库并准备实例
                if (writeVector) {
                    const targetIndex = await getIndex(collectionId, true);
                    const stats = await targetIndex.listItems();
                    if (stats.length > 0) {
                        console.log(`[Anima KB] 发现旧向量数据，正在重建库...`);
                        const folderPath = path.join(VECTOR_ROOT, collectionId);
                        if (fs.existsSync(folderPath)) {
                            fs.rmSync(folderPath, {
                                recursive: true,
                                force: true,
                            });
                            activeIndexes.delete(collectionId);
                        }
                    }
                    cleanIndex = await getIndex(collectionId, true);
                }

                // 4. 切片
                const chunks = chunkText(fileContent, {
                    delimiter: settings.delimiter,
                    chunkSize: settings.chunk_size,
                });

                console.log(
                    `[Anima KB] 切片完成，共 ${chunks.length} 个片段。开始处理...`,
                );
                const bm25Chunks = [];

                // 5. 循环处理切片
                for (let i = 0; i < chunks.length; i++) {
                    const chunkText = chunks[i];

                    // 🌟 默认生成一个随机 ID (如果不开向量库，BM25 依然需要 ID 才能运作)
                    let documentId = `chunk_${i}_${Date.now()}`;

                    // 🌟 如果开启了向量化，调用模型 API
                    if (writeVector) {
                        try {
                            const vector = await getEmbedding(
                                chunkText,
                                apiConfig,
                            );
                            const insertedItem = await cleanIndex.insertItem({
                                vector: vector,
                                metadata: {
                                    text: chunkText,
                                    doc_name: fileName,
                                    source_type: "knowledge",
                                    chunk_index: i,
                                    timestamp: Date.now(),
                                },
                            });
                            // 替换为真实的向量库 UUID
                            documentId = insertedItem.id;

                            if ((i + 1) % 10 === 0)
                                console.log(
                                    `[Anima KB] 向量化进度: ${i + 1}/${chunks.length}`,
                                );
                        } catch (err) {
                            console.error(
                                `[Anima KB] 片段 ${i} 向量化失败:`,
                                err.message,
                            );
                        }
                    }

                    // 无论是否生成了向量，都把文本和确定的 ID 存起来给 BM25 备用
                    bm25Chunks.push({
                        id: documentId,
                        text: chunkText,
                        chunk_index: i,
                        doc_name: fileName,
                        timestamp: Date.now(),
                    });
                }

                // 6. 如果开启了 BM25，执行 BM25 构建
                if (bm25Config && bm25Config.enabled && bm25Chunks.length > 0) {
                    try {
                        const dict = bm25Config.dictionary || [];
                        // 覆盖原有的 BM25 库
                        await bm25Engine.deleteIndex(collectionId);
                        await bm25Engine.buildIndex(
                            collectionId,
                            bm25Chunks,
                            dict,
                            "kb",
                        );
                        console.log(`[Anima KB] BM25 索引构建完成！`);
                    } catch (bm25Err) {
                        console.error(
                            `[Anima KB BM25] ❌ 构建索引失败:`,
                            bm25Err,
                        );
                    }
                }
            });

            res.json({ success: true, collectionId: collectionId, count: 0 });
        } catch (err) {
            console.error("[Anima KB Error]", err);
            res.status(500).send(err.message);
        }
    });

    router.get("/list", async (req, res) => {
        try {
            if (!fs.existsSync(VECTOR_ROOT)) {
                return res.json([]);
            }
            // 读取 vectors 文件夹下的所有文件夹名称
            const files = fs.readdirSync(VECTOR_ROOT, { withFileTypes: true });
            const dirs = files
                .filter((dirent) => dirent.isDirectory())
                .map((dirent) => dirent.name);
            res.json(dirs);
        } catch (err) {
            res.status(500).send(err.message);
        }
    });

    router.get("/bm25/list", async (req, res) => {
        try {
            const bm25Root = path.join(__dirname, "data", "bm25_indexes");
            if (!fs.existsSync(bm25Root)) {
                return res.json([]); // 如果目录不存在，返回空列表
            }
            // 读取目录下的所有文件
            const files = fs.readdirSync(bm25Root, { withFileTypes: true });
            const libs = files
                .filter(
                    (dirent) =>
                        dirent.isFile() && dirent.name.endsWith(".json"),
                )
                .map((dirent) => dirent.name.replace(".json", "")); // 去掉后缀，只留库名

            res.json(libs);
        } catch (err) {
            console.error(`[Anima BM25] 读取库列表失败: ${err.message}`);
            res.status(500).send(err.message);
        }
    });

    // ==========================================
    // 🟢 新增：导出单个 BM25 库文件
    // ==========================================
    router.post("/bm25/export_single", async (req, res) => {
        const { libName } = req.body;
        try {
            const safeName = libName.replace(
                /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                "_",
            );
            const filePath = path.join(
                __dirname,
                "data",
                "bm25_indexes",
                `${safeName}.json`,
            );

            if (!fs.existsSync(filePath)) {
                return res
                    .status(404)
                    .json({ success: false, message: "未找到该库的实体文件" });
            }

            const data = fs.readFileSync(filePath, "utf-8");
            res.json({ success: true, data: JSON.parse(data) });
        } catch (e) {
            console.error(`[Anima BM25] 导出库失败: ${e.message}`);
            res.status(500).json({ success: false, message: e.message });
        }
    });

    // ==========================================
    // 🟢 新增：导入单个 BM25 库文件
    // ==========================================
    router.post("/bm25/import_single", async (req, res) => {
        const { libName, data } = req.body;
        try {
            const safeName = libName.replace(
                /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                "_",
            );
            const bm25Root = path.join(__dirname, "data", "bm25_indexes");

            if (!fs.existsSync(bm25Root)) {
                fs.mkdirSync(bm25Root, { recursive: true });
            }

            const filePath = path.join(bm25Root, `${safeName}.json`);
            fs.writeFileSync(filePath, JSON.stringify(data));

            console.log(`[Anima BM25] 📥 成功导入库: ${safeName}`);
            res.json({ success: true });
        } catch (e) {
            console.error(`[Anima BM25] 导入库失败: ${e.message}`);
            res.status(500).json({ success: false, message: e.message });
        }
    });

    // ==========================================
    // 🟢 新增：物理删除单个 BM25 库
    // ==========================================
    router.post("/bm25/delete_single", async (req, res) => {
        const { libName } = req.body;
        try {
            const safeName = libName.replace(
                /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                "_",
            );
            // 直接调用你写好的 bm25Engine 物理删除引擎
            await bm25Engine.deleteIndex(safeName);
            res.json({ success: true });
        } catch (e) {
            console.error(`[Anima BM25] 删除库失败: ${e.message}`);
            res.status(500).json({ success: false, message: e.message });
        }
    });

    // ==========================================
    // 🟢 新增：从 BM25 逆向重构向量库
    // ==========================================
    router.post("/rebuild_vector_from_bm25", async (req, res) => {
        const { collectionId, apiConfig } = req.body;
        if (!collectionId) return res.status(400).send("Missing collectionId");
        if (!apiConfig || !apiConfig.key)
            return res.status(400).send("Missing API Config");

        const safeName = collectionId.replace(
            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
            "_",
        );
        console.log(`[Anima RAG] 🚀 尝试从 BM25 逆向重构向量库: ${safeName}`);

        try {
            await runInQueue(safeName, async () => {
                // 1. 读取底层的 BM25 库文件
                const bm25Path = path.join(
                    __dirname,
                    "data",
                    "bm25_indexes",
                    `${safeName}.json`,
                );
                if (!fs.existsSync(bm25Path)) {
                    throw new Error("BM25 库文件不存在，无法提取原文");
                }

                // MiniSearch 导出的 JSON 中，storedFields 保存了完整的原始信息
                const indexData = JSON.parse(
                    fs.readFileSync(bm25Path, "utf-8"),
                );
                const allDocs = Object.values(indexData.storedFields || {});

                if (allDocs.length === 0) {
                    throw new Error("BM25 库中未检测到存储的文本内容");
                }

                // 2. 初始化目标向量库 (如果旧库有残留碎片，先安全清空)
                const targetIndex = await getIndex(safeName, true);
                const existingStats = await targetIndex.listItems();
                if (existingStats.length > 0) {
                    const folderPath = path.join(VECTOR_ROOT, safeName);
                    if (fs.existsSync(folderPath)) {
                        fs.rmSync(folderPath, { recursive: true, force: true });
                        activeIndexes.delete(safeName);
                    }
                }

                // 获取彻底纯净的向量库实例
                const cleanIndex = await getIndex(safeName, true);

                let successCount = 0;
                let failCount = 0;

                // 3. 循环遍历提取的文本，请求 API 生成向量
                for (const doc of allDocs) {
                    if (!doc.text) continue;
                    try {
                        const vector = await getEmbedding(doc.text, apiConfig);

                        // 插入向量库，完美继承当初在 BM25 记录的所有切片特征
                        await cleanIndex.insertItem({
                            vector: vector,
                            metadata: {
                                text: doc.text,
                                doc_name: doc.doc_name || "recovered_from_bm25",
                                source_type: "knowledge",
                                chunk_index: doc.chunk_index || 0,
                                timestamp: doc.timestamp || Date.now(),
                                tags: doc.tags || [],
                                index: doc.index,
                                batch_id: doc.batch_id,
                            },
                        });

                        successCount++;
                        // 终端每10条打印一次进度
                        if (successCount % 10 === 0) {
                            console.log(
                                `[Anima RAG] 逆向向量化进度: ${successCount}/${allDocs.length}`,
                            );
                        }
                    } catch (err) {
                        console.error(
                            `[Anima RAG] 逆向向量化失败 (${doc.id}): ${err.message}`,
                        );
                        failCount++;
                    }
                }

                console.log(
                    `[Anima RAG] ✅ 逆向重构向量库完成. 成功: ${successCount}, 失败: ${failCount}`,
                );
                res.json({
                    success: true,
                    count: successCount,
                    failed: failCount,
                });
            });
        } catch (err) {
            console.error(`[Anima RAG] 逆向重构失败: ${err.message}`);
            res.status(500).send(err.message);
        }
    });

    // ==========================================
    // 🟢 新增/修改：全量重建单个 BM25 库 (支持无向量库兜底)
    // ==========================================
    router.post("/bm25/rebuild_collection", async (req, res) => {
        const { collectionId, bm25Config } = req.body;
        if (!collectionId)
            return res
                .status(400)
                .json({ success: false, message: "Missing collectionId" });

        const safeName = collectionId.replace(
            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
            "_",
        );
        console.log(`[Anima BM25] 🚀 开始全量重建库: ${safeName}`);

        try {
            await runInQueue(safeName, async () => {
                const bm25Chunks = [];
                let hasVectorSource = false;

                // 1. 优先尝试获取向量库的数据作为原文来源
                try {
                    const targetIndex = await getIndex(safeName, false);
                    if (targetIndex) {
                        hasVectorSource = true;
                        const items = await targetIndex.listItems();
                        const folderPath = path.join(VECTOR_ROOT, safeName);

                        for (const item of items) {
                            try {
                                const fileName =
                                    item.metadataFile || `${item.id}.json`;
                                const filePath = path.join(
                                    folderPath,
                                    fileName,
                                );
                                if (!fs.existsSync(filePath)) continue;

                                const fileContent = await fs.promises.readFile(
                                    filePath,
                                    "utf-8",
                                );
                                const fullData = JSON.parse(fileContent);
                                const meta = fullData.metadata || fullData;

                                if (!meta.text) continue;

                                bm25Chunks.push({
                                    id: item.id,
                                    text: meta.text,
                                    tags: meta.tags || [],
                                    timestamp: meta.timestamp || Date.now(),
                                    index: meta.index,
                                    batch_id: meta.batch_id,
                                    chunk_index: meta.chunk_index,
                                    doc_name: meta.doc_name,
                                });
                            } catch (readErr) {
                                console.warn(
                                    `[Anima BM25] 读取向量库条目 ${item.id} 失败，跳过`,
                                );
                            }
                        }
                    }
                } catch (e) {
                    // 忽略报错，交给下方的兜底逻辑处理
                }

                // 2. 🛡️ 兜底逻辑：如果向量库不存在，从现存的旧 BM25 库中提取文本
                if (!hasVectorSource || bm25Chunks.length === 0) {
                    console.log(
                        `[Anima BM25] ⚠️ 未检测到向量库，尝试从旧 BM25 库提取原文...`,
                    );
                    const bm25Path = path.join(
                        __dirname,
                        "data",
                        "bm25_indexes",
                        `${safeName}.json`,
                    );

                    if (fs.existsSync(bm25Path)) {
                        const indexData = JSON.parse(
                            fs.readFileSync(bm25Path, "utf-8"),
                        );
                        const allDocs = Object.values(
                            indexData.storedFields || {},
                        );

                        for (const doc of allDocs) {
                            if (!doc.text) continue;
                            bm25Chunks.push({
                                id: doc.id,
                                text: doc.text,
                                tags: doc.tags || [],
                                timestamp: doc.timestamp || Date.now(),
                                index: doc.index,
                                batch_id: doc.batch_id,
                                chunk_index: doc.chunk_index,
                                doc_name: doc.doc_name,
                            });
                        }
                    } else {
                        throw new Error(
                            "找不到对应的底层向量库提供文本数据，且旧 BM25 库也不存在。",
                        );
                    }
                }

                // 3. 🛡️ 核心保障：物理删除旧的 BM25 库（绝对防重复！）
                await bm25Engine.deleteIndex(safeName);

                // 4. 重新构建全新的 BM25 库
                if (bm25Chunks.length > 0) {
                    const dict = bm25Config?.dictionary || [];
                    await bm25Engine.buildIndex(
                        safeName,
                        bm25Chunks,
                        dict,
                        "chat",
                    );
                }

                console.log(
                    `[Anima BM25] ✅ 库 ${safeName} 重建完毕，共写入 ${bm25Chunks.length} 条数据`,
                );
                res.json({ success: true, count: bm25Chunks.length });
            });
        } catch (err) {
            console.error(`[Anima BM25] 重建失败: ${err.message}`);
            res.status(500).json({ success: false, message: err.message });
        }
    });

    // ==========================================
    // 🟢 新增：单切片 BM25 增量重构接口
    // ==========================================
    router.post("/bm25/rebuild_slice", async (req, res) => {
        const {
            collectionId,
            index,
            text,
            tags,
            timestamp,
            batch_id,
            bm25Config,
        } = req.body;

        if (!collectionId || index === undefined || !text) {
            return res
                .status(400)
                .json({ success: false, message: "Missing required fields" });
        }

        try {
            const safeName = collectionId.replace(
                /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                "_",
            );

            // 尝试去向量库中找原有的 document ID (防止和向量库ID脱节)，如果没有向量库，就兜底使用 index 作为 ID
            let documentId = `slice_${index}`;
            try {
                const targetIndex = await getIndex(safeName, false);
                if (targetIndex) {
                    const allItems = await targetIndex.listItems();
                    const target = allItems.find(
                        (item) =>
                            item.metadata &&
                            String(item.metadata.index) === String(index),
                    );
                    if (target) documentId = target.id;
                }
            } catch (e) {
                // 忽略：向量库可能还没创建，BM25 允许独立运行
            }

            const dict = bm25Config.dictionary || [];

            // 呼叫底层的 BM25 引擎进行写入
            await bm25Engine.upsertDocument(
                safeName,
                { id: documentId, text, tags, timestamp, index, batch_id },
                dict,
                "chat",
            );

            res.json({ success: true });
        } catch (err) {
            console.error(`[Anima BM25] 切片重构失败: ${err.message}`);
            res.status(500).json({ success: false, message: err.message });
        }
    });

    // 🚀 新增：BM25 批量增量重构接口
    router.post("/bm25/rebuild_slice_batch", async (req, res) => {
        try {
            const { collectionId, slices, bm25Config } = req.body;

            if (!slices || !Array.isArray(slices) || slices.length === 0) {
                return res.json({ success: false, message: "切片数组为空" });
            }

            const addedCount = await bm25Engine.buildIndexBatch(
                collectionId,
                slices,
                bm25Config,
            );

            res.json({ success: true, count: addedCount });
        } catch (e) {
            console.error("[Anima BM25] 批量重构路由异常:", e);
            res.json({ success: false, message: e.message });
        }
    });

    router.post("/view_collection", async (req, res) => {
        const { collectionId } = req.body;
        if (!collectionId) return res.status(400).send("Missing collectionId");

        try {
            // 准备路径工具
            const safeName = collectionId.replace(
                /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                "_",
            );
            const collectionPath = path.join(VECTOR_ROOT, safeName);

            let validItems = [];

            // 1. 尝试获取向量索引实例
            const targetIndex = await getIndex(collectionId, false);

            if (targetIndex) {
                const indexItems = await targetIndex.listItems();
                const formattedItems = await Promise.all(
                    indexItems.map(async (entry) => {
                        try {
                            const fileName =
                                entry.metadataFile || `${entry.id}.json`;
                            const filePath = path.join(
                                collectionPath,
                                fileName,
                            );

                            if (!fs.existsSync(filePath)) return null;

                            const fileContent = await fs.promises.readFile(
                                filePath,
                                "utf-8",
                            );
                            const fullData = JSON.parse(fileContent);
                            const meta = fullData.metadata || fullData;

                            return {
                                id: entry.id,
                                text: meta.text || "",
                                metadata: {
                                    chunk_index: meta.chunk_index,
                                    doc_name: meta.doc_name,
                                    timestamp: meta.timestamp,
                                },
                            };
                        } catch (e) {
                            return null;
                        }
                    }),
                );
                validItems = formattedItems.filter((i) => i !== null);
            }

            // 2. 🟢 核心修复：如果向量库不存在（或被清空），尝试兜底读取同名的 BM25 本地文件
            if (validItems.length === 0) {
                const bm25Path = path.join(
                    __dirname,
                    "data",
                    "bm25_indexes",
                    `${safeName}.json`,
                );
                if (fs.existsSync(bm25Path)) {
                    const indexData = JSON.parse(
                        fs.readFileSync(bm25Path, "utf-8"),
                    );
                    const allDocs = Object.values(indexData.storedFields || {});

                    validItems = allDocs.map((doc) => ({
                        id: doc.id,
                        text: doc.text || "",
                        metadata: {
                            chunk_index: doc.chunk_index,
                            doc_name: doc.doc_name,
                            timestamp: doc.timestamp,
                        },
                    }));
                }
            }

            // 3. 如果两边都找不到，才抛出 404
            if (validItems.length === 0) {
                return res
                    .status(404)
                    .json({ error: "Database not found or is empty" });
            }

            res.json({ items: validItems });
        } catch (err) {
            console.error(`[Anima RAG] View Collection Error: ${err.message}`);
            res.status(500).send(err.message);
        }
    });

    // ==========================================
    // 🔍 改造后的查询接口 (支持并行双轨检索)
    // ==========================================
    router.post("/query", async (req, res) => {
        try {
            const {
                searchText,
                bm25SearchText,
                apiConfig,
                ignore_ids,
                echoConfig,
                sessionId,
                is_swipe,
                rerankConfig,
                bm25Configs = {},
            } = req.body;

            // --- 兼容旧版参数 ---
            const legacyCollectionIds = req.body.collectionIds;
            const legacyStrategy = req.body.strategy;

            // --- 新版参数 ---
            const chatContext = req.body.chatContext || {
                ids: legacyCollectionIds,
                strategy: legacyStrategy,
            };
            const kbContext = req.body.kbContext || { ids: [], strategy: null };

            // 2. 向量化
            if (!searchText && !bm25SearchText)
                return res.json({ chat_results: [], kb_results: [] });

            // 兜底：如果向量检索词为空，则 vector 为 null，防止 getEmbedding 报错
            let vector = null;
            if (searchText) {
                vector = await getEmbedding(searchText, apiConfig);
            }

            // ============================================================
            // 🧠 [新增] 会话状态预处理 (GC vs Resurrection)
            // ============================================================
            // 🛠️ 修复点：初始化为对象 {} 而不是数组 []，避免类型冲突
            let sessionData = { memories: {} };
            let lastMemories = {};

            if (sessionId) {
                // 读取 Session
                const loaded = loadSession(sessionId);
                // 确保 memories 存在，且如果是数组(旧数据)要转为对象，如果是对象则直接用
                if (Array.isArray(loaded.memories)) {
                    // 兼容旧数据的兜底逻辑：把数组转为 ID Map
                    loaded.memories.forEach((m) => {
                        if (m && m.item && m.item.id)
                            lastMemories[m.item.id] = m;
                    });
                } else {
                    lastMemories = loaded.memories || {};
                }

                // --- 核心逻辑开始 ---
                if (is_swipe) {
                    // 🅰️ 【Swipe 模式】：亡者复苏
                    let resurrectionCount = 0;
                    for (const [key, mem] of Object.entries(lastMemories)) {
                        if (mem.life <= 0) {
                            mem.life = 1; // 临时复活
                            resurrectionCount++;
                        }
                    }
                    if (resurrectionCount > 0) {
                        console.log(
                            `[Anima Echo] 🔄 检测到 Swipe: 临时复活了 ${resurrectionCount} 条僵尸记忆`,
                        );
                    }
                } else {
                    // 🅱️ 【Normal 模式】：垃圾回收 (GC)
                    const livingMemories = {};
                    let gcCount = 0;
                    for (const [key, mem] of Object.entries(lastMemories)) {
                        if (mem.life > 0) {
                            livingMemories[key] = mem;
                        } else {
                            gcCount++;
                        }
                    }
                    if (gcCount > 0) {
                        console.log(
                            `[Anima Echo] 🧹 新对话开始: 清理了 ${gcCount} 条已枯竭的记忆`,
                        );
                        lastMemories = livingMemories;
                    }
                }

                // 🛠️ 修复点：赋值回 sessionData，此时类型匹配了 (都是对象)
                sessionData.memories = lastMemories;
            }

            // 3. 定义并行任务
            const tasks = [];

            let bm25ChatResults = [];
            let bm25KbResults = [];

            // --- 任务 A: 聊天记录检索 ---
            const chatTask = async () => {
                if (!vector) return [];
                const targetIds = Array.isArray(chatContext.ids)
                    ? chatContext.ids.filter((id) => id)
                    : [];
                if (targetIds.length === 0) return [];

                const rawIndices = (
                    await Promise.all(
                        targetIds.map((id) =>
                            getIndex(id, false).catch((e) => {
                                // 🚨 抓捕幽灵报错：把底层的崩溃原因打印出来
                                console.error(
                                    `[Anima 致命抓捕] 加载库 ${id} 时底层崩溃:`,
                                    e,
                                );
                                return null;
                            }),
                        ),
                    )
                ).filter((i) => i !== null);

                const uniqueIndices = [...new Set(rawIndices)];
                if (uniqueIndices.length === 0) return [];

                const safeIgnoreIds = Array.isArray(ignore_ids)
                    ? ignore_ids
                    : [];
                const strat = chatContext.strategy;

                if (strat && strat.enabled) {
                    strat.searchText = searchText;
                    strat.rerankConfig = rerankConfig;

                    return await performDynamicStrategy(
                        uniqueIndices,
                        vector,
                        strat, // strat 里现在包含了 searchText 和 rerankConfig
                        safeIgnoreIds,
                    );
                } else {
                    // 简单模式
                    const simpleCount =
                        strat?.steps?.find((s) => s.type === "base")?.count ||
                        5;
                    const minScore = strat?.min_score || 0;
                    const simpleFilter =
                        safeIgnoreIds.length > 0
                            ? { index: { $nin: safeIgnoreIds } }
                            : null;

                    // 🟢 [新增] 获取参数
                    const recentWeight = strat?.recent_weight || 0;
                    const currentSessionId = strat?.current_session_id || null;

                    let raw = await queryMultiIndices(
                        uniqueIndices,
                        vector,
                        simpleCount * 1.5,
                        simpleFilter,
                        "SimpleChat",
                        recentWeight, // 🟢 [新增透传]
                        currentSessionId, // 🟢 [新增透传]
                    );
                    raw["_debug_logs"] = raw["_debug_logs"] || [];
                    raw["_debug_logs"].push({
                        step: "Base",
                        library: "Simple",
                        score: 0,
                        tags: "No Strategy",
                    });
                    raw = raw
                        .filter((r) => r.score >= minScore)
                        .slice(0, simpleCount);
                    raw.sort((a, b) => {
                        const timeA = new Date(
                            a.item.metadata.timestamp || 0,
                        ).getTime();
                        const timeB = new Date(
                            b.item.metadata.timestamp || 0,
                        ).getTime();
                        return timeA - timeB;
                    });
                    return raw;
                }
            };
            tasks.push(chatTask());

            // --- 任务 B: 知识库检索 ---
            const kbTask = async () => {
                const targetIds = Array.isArray(kbContext.ids)
                    ? kbContext.ids.filter((id) => id)
                    : [];
                if (targetIds.length === 0) return [];

                const rawIndices = (
                    await Promise.all(
                        targetIds.map((id) =>
                            getIndex(id, false).catch(() => null),
                        ),
                    )
                ).filter((i) => i !== null);

                const uniqueIndices = [...new Set(rawIndices)];
                if (uniqueIndices.length === 0) return [];

                const strat = kbContext.strategy || { min_score: 0.5 };
                const simpleCount = strat.search_top_k || 3;
                const minScore = strat.min_score || 0.5;

                // 2. 执行检索 (修改点：移除 * 2)
                // queryMultiIndices 的逻辑是：从“每个”库里都取 simpleCount 条
                // 然后聚合、排序，最后截取前 simpleCount 条
                // 这完美符合你的要求：N -> N
                let raw = await queryMultiIndices(
                    uniqueIndices,
                    vector,
                    simpleCount,
                    null,
                    "KB",
                );

                raw = raw
                    .filter((r) => r.score >= minScore)
                    .slice(0, simpleCount);

                return raw;
            };
            tasks.push(kbTask());

            // 🟢 [新增] 任务 C: BM25 聊天库检索
            const bm25ChatTask = async () => {
                if (!bm25Configs.chat || bm25Configs.chat.length === 0) return;

                const validBm25Text =
                    bm25SearchText && bm25SearchText.trim().length > 0
                        ? bm25SearchText
                        : null;
                const targetText = validBm25Text || searchText;
                if (!targetText || targetText.trim() === "") return;

                // =========================================================
                // ✨ 核心改造 1：精准剥离“最新 User 意图”与“历史上下文” (强力抗干扰版)
                // =========================================================
                let userTextPool = "";
                let contextTextPool = "";

                // 统一使用物理截断，防止没有换行符的合并文本干扰
                const lowerTargetText = targetText.toLowerCase();
                const lastUserIdx = lowerTargetText.lastIndexOf("user:");

                if (lastUserIdx !== -1) {
                    contextTextPool = targetText.substring(0, lastUserIdx);
                    userTextPool = targetText.substring(lastUserIdx);
                } else {
                    userTextPool = targetText;
                }

                // =========================================================
                // ✨ 核心改造 1.5：意图雷达 (探测时间极值)
                // =========================================================
                const intentFirstWords = [
                    "第一次",
                    "第一回",
                    "第一眼",
                    "第一面",
                    "首次",
                    "首回",
                    "初回",
                    "初次",
                    "最早",
                    "最初",
                ];
                const intentLastWords = [
                    "最后",
                    "最近",
                    "上次",
                    "上一次",
                    "上一回",
                    "上回",
                ];

                let temporalIntent = null;
                const lowerUserTextOnly = userTextPool.toLowerCase();

                if (
                    intentFirstWords.some((w) => lowerUserTextOnly.includes(w))
                ) {
                    temporalIntent = "first";
                } else if (
                    intentLastWords.some((w) => lowerUserTextOnly.includes(w))
                ) {
                    temporalIntent = "last";
                }

                // =========================================================
                // ✨ 核心改造 2：分层扫描与收集
                // =========================================================
                let userTriggeredIndexes = [];
                let contextTriggeredIndexes = [];
                let hasTrigger = false;
                let totalRules = 0;

                const lowerUserText = userTextPool.toLowerCase();
                const lowerContextText = contextTextPool.toLowerCase();

                bm25Configs.chat.forEach((config) => {
                    const dict =
                        config.dictionary || config.dict || config.words || [];
                    totalRules += dict.length;

                    dict.forEach((rule) => {
                        const rawTrigger = rule.trigger || "";
                        const triggers = rawTrigger
                            .split(/[,，]/)
                            .map((t) => t.trim())
                            .filter(Boolean);
                        const indexWord = (rule.index || "").trim();

                        const actualTriggers = [...triggers];
                        if (indexWord && !actualTriggers.includes(indexWord))
                            actualTriggers.push(indexWord);

                        if (actualTriggers.length > 0) {
                            const hitUser = actualTriggers.some((t) =>
                                lowerUserText.includes(t.toLowerCase()),
                            );
                            if (hitUser) {
                                hasTrigger = true;
                                if (indexWord)
                                    userTriggeredIndexes.push(indexWord);
                            }
                            if (!hitUser) {
                                const hitContext = actualTriggers.some((t) =>
                                    lowerContextText.includes(t.toLowerCase()),
                                );
                                if (hitContext) {
                                    hasTrigger = true;
                                    if (indexWord)
                                        contextTriggeredIndexes.push(indexWord);
                                }
                            }
                        }
                    });
                });

                if (totalRules > 0 && !hasTrigger) {
                    console.log(
                        `[Anima BM25] 🛑 未命中任何触发词，跳过 Chat 检索。`,
                    );
                    return;
                }

                // =========================================================
                // ✨ 核心改造 3：阶梯式词频加权 (TF Boosting) & Debug 日志
                // =========================================================
                const chatTopK = bm25Configs.chat_top_k || 3;
                let intentResults = [];

                // 🌟 线路 A：如果探测到时间极值意图，交由特种部队处理
                // 🔴 核心修复：把 User 和 Context 里的实体合并！因为 User 经常用代词省略主语
                if (temporalIntent && userTriggeredIndexes.length > 0) {
                    const uniqueUserEntities = [
                        ...new Set(userTriggeredIndexes),
                    ];

                    console.log(
                        `[Anima BM25 Debug] ⏱️ 探测到时间极值意图: [${temporalIntent}] | 关联核心实体(仅User): [${uniqueUserEntities.join(", ")}]`,
                    );

                    intentResults = await bm25Engine.temporalIntentSearch(
                        uniqueUserEntities,
                        bm25Configs.chat,
                        temporalIntent,
                        ignore_ids,
                    );

                    intentResults = intentResults.map((r) => {
                        r.score = 999.0;
                        r._is_intent = true;
                        return r;
                    });

                    console.log(
                        `[Anima BM25 Debug] ⚡ 意图拦截执行完毕，提取了 ${intentResults.length} 条绝对时间切片。`,
                    );
                }

                // 计算剩余的 Top K 坑位
                const remainingK = Math.max(0, chatTopK - intentResults.length);
                let standardResults = [];

                // 🌟 线路 B：剩余坑位交由传统的 TF-IDF 模糊联想填充
                if (remainingK > 0) {
                    const cleanText = targetText
                        .replace(/[^\u4e00-\u9fa5a-zA-Z0-9]/g, " ")
                        .replace(/\s+/g, " ")
                        .trim();
                    let boostStrArr = [];

                    if (contextTriggeredIndexes.length > 0) {
                        const uniqueContexts = [
                            ...new Set(contextTriggeredIndexes),
                        ];
                        boostStrArr.push(uniqueContexts.join(" "));
                    }

                    if (userTriggeredIndexes.length > 0) {
                        const uniqueUsers = [...new Set(userTriggeredIndexes)];
                        boostStrArr.push(
                            uniqueUsers.map((t) => `${t} ${t} ${t}`).join(" "),
                        );
                    }

                    const finalBoostStr = boostStrArr.join(" ");
                    const boostedQuery = finalBoostStr
                        ? `${cleanText} ${finalBoostStr}`
                        : cleanText;

                    console.log(
                        `[Anima BM25 Debug] 🚀 发往引擎的最终加权检索词 (Boosted Query):\n=> "${boostedQuery}"`,
                    );

                    standardResults = await bm25Engine.searchPipeline(
                        boostedQuery,
                        bm25Configs.chat,
                        remainingK,
                        "chat",
                        ignore_ids,
                    );
                    console.log(
                        `[Anima BM25 Debug] 📊 常规模糊检索执行完毕，返回了 ${standardResults.length} 条结果。`,
                    );
                }

                // 🌟 合并并去重
                const mergedMap = new Map();
                [...intentResults, ...standardResults].forEach((item) => {
                    const uniqueKey = item.index || item.id;
                    if (!mergedMap.has(uniqueKey)) {
                        mergedMap.set(uniqueKey, item);
                    }
                });

                const combinedResults = Array.from(mergedMap.values());

                // 解决前端显示 Unknown 数据库的问题
                bm25ChatResults = combinedResults.map((r) => {
                    const src =
                        r._source_db ||
                        r.dbId ||
                        r.collectionId ||
                        r._source_collection ||
                        r.source ||
                        "Unknown";
                    return {
                        ...r,
                        type: "bm25",
                        source: src,
                        _source_collection: src,
                    };
                });
            };
            tasks.push(bm25ChatTask());

            // 🟢 [新增] 任务 D: BM25 知识库检索
            const bm25KbTask = async () => {
                if (!bm25Configs.kb || bm25Configs.kb.length === 0) return;

                const validBm25Text =
                    bm25SearchText && bm25SearchText.trim().length > 0
                        ? bm25SearchText
                        : null;
                const targetText = validBm25Text || searchText;
                if (!targetText || targetText.trim() === "") return;

                // =========================================================
                // ✨ 核心改造 1：精准剥离“最新 User 意图”与“历史上下文”
                // =========================================================
                let userTextPool = "";
                let contextTextPool = "";

                const lines = targetText.split("\n");
                let lastUserIdx = -1;

                // 1. 倒序查找，精准定位“最后一楼 User”所在的行索引
                for (let i = lines.length - 1; i >= 0; i--) {
                    if (lines[i].trim().toLowerCase().startsWith("user:")) {
                        lastUserIdx = i;
                        break;
                    }
                }

                if (lastUserIdx !== -1) {
                    // 2. 将最后一楼 User 之前的所有行，全部归入辅助上下文
                    contextTextPool = lines.slice(0, lastUserIdx).join(" ");
                    // 3. 将最后一楼 User 及其之后的多行，归入强意图池
                    userTextPool = lines.slice(lastUserIdx).join(" ");
                } else {
                    // 兜底：如果没有 user: 前缀，统统算作强意图
                    userTextPool = targetText;
                }

                // =========================================================
                // ✨ 核心改造 2：分层扫描与收集
                // =========================================================
                let userTriggeredIndexes = [];
                let contextTriggeredIndexes = [];
                let hasTrigger = false;
                let totalRules = 0;

                const lowerUserText = userTextPool.toLowerCase();
                const lowerContextText = contextTextPool.toLowerCase();

                bm25Configs.kb.forEach((config) => {
                    const dict =
                        config.dictionary || config.dict || config.words || [];
                    totalRules += dict.length;

                    dict.forEach((rule) => {
                        const rawTrigger = rule.trigger || "";
                        const triggers = rawTrigger
                            .split(/[,，]/)
                            .map((t) => t.trim())
                            .filter(Boolean);
                        const indexWord = (rule.index || "").trim();

                        const actualTriggers = [...triggers];
                        if (indexWord && !actualTriggers.includes(indexWord)) {
                            actualTriggers.push(indexWord);
                        }

                        if (actualTriggers.length > 0) {
                            // 先扫描 User 强意图池
                            const hitUser = actualTriggers.some((t) =>
                                lowerUserText.includes(t.toLowerCase()),
                            );

                            if (hitUser) {
                                hasTrigger = true;
                                if (indexWord)
                                    userTriggeredIndexes.push(indexWord);
                            }

                            // 如果 User 没命中，再看历史上下文有没有命中
                            if (!hitUser) {
                                const hitContext = actualTriggers.some((t) =>
                                    lowerContextText.includes(t.toLowerCase()),
                                );
                                if (hitContext) {
                                    hasTrigger = true;
                                    if (indexWord)
                                        contextTriggeredIndexes.push(indexWord);
                                }
                            }
                        }
                    });
                });

                // 如果配置了知识库词典但全都没命中，跳过检索
                if (totalRules > 0 && !hasTrigger) {
                    console.log(
                        `[Anima BM25] 🛑 未命中任何触发词，跳过 KB 检索。`,
                    );
                    return;
                }

                // =========================================================
                // ✨ 核心改造 3：阶梯式词频加权 (TF Boosting)
                // =========================================================
                const cleanText = targetText
                    .replace(/[^\u4e00-\u9fa5a-zA-Z0-9]/g, " ")
                    .replace(/\s+/g, " ")
                    .trim();

                let boostStrArr = [];

                // 👉 辅助内容 (历史楼层)：重复 1 次
                if (contextTriggeredIndexes.length > 0) {
                    const uniqueContexts = [
                        ...new Set(contextTriggeredIndexes),
                    ];
                    boostStrArr.push(uniqueContexts.join(" "));
                }

                // 👉 核心意图 (最新 User)：重复 3 次
                if (userTriggeredIndexes.length > 0) {
                    const uniqueUsers = [...new Set(userTriggeredIndexes)];
                    const userBoost = uniqueUsers
                        .map((t) => `${t} ${t} ${t}`)
                        .join(" ");
                    boostStrArr.push(userBoost);
                    console.log(
                        `[Anima KB BM25] 🎯 强意图命中(User)! 加权: [${uniqueUsers.join(", ")}]`,
                    );
                }

                const finalBoostStr = boostStrArr.join(" ");
                const boostedQuery = finalBoostStr
                    ? `${cleanText} ${finalBoostStr}`
                    : cleanText;

                const strat = kbContext.strategy || {};
                const bm25Count = strat.bm25_top_k || 3;

                const results = await bm25Engine.searchPipeline(
                    boostedQuery,
                    bm25Configs.kb,
                    bm25Count,
                    "kb",
                );

                bm25KbResults = results.map((r) => {
                    const src =
                        r._source_db ||
                        r.dbId ||
                        r.collectionId ||
                        r._source_collection ||
                        r.source ||
                        "Unknown";
                    return {
                        ...r,
                        type: "bm25",
                        source: src,
                        _source_collection: src,
                    };
                });
            };
            tasks.push(bm25KbTask());

            // 4. 并行等待结果
            const [chatRaw, kbRaw] = await Promise.all(tasks);
            const collectedLogs =
                chatRaw && chatRaw["_debug_logs"] ? chatRaw["_debug_logs"] : [];
            // ============================================================
            // 🧠 [新增] 回响机制集成 (Echo Mechanism Integration)
            // ============================================================
            let finalChatResults = chatRaw || [];

            if (sessionId && chatContext.ids && chatContext.ids.length > 0) {
                try {
                    console.log(
                        `[Anima Echo] 🧠 启动回响处理... Session: ${sessionId}`,
                    );

                    // 🛠️ 修复点：直接使用预处理好的 lastMemories (含复活/GC后的状态)
                    // 不要再调用 loadSession 了

                    // 获取全局校验池
                    /* 
                    const targetIds = chatContext.ids.filter((id) => id);
                    const rawIndices = await Promise.all(
                        targetIds.map((id) =>
                            getIndex(id, false).catch(() => null),
                        ),
                    );
                    const uniqueIndices = [...new Set(rawIndices)].filter(
                        (i) => i !== null,
                    );

                    const globalTop50 = await queryMultiIndices(
                        uniqueIndices,
                        vector,
                        50,
                        null,
                        "EchoValidator",
                    );
                    */
                    const globalTop50 = chatRaw["_echo_pool"] || [];

                    console.log(
                        `[Anima Echo] ♻️ 复用 Base 检索池: ${globalTop50.length} 条候选`,
                    );

                    // 3. 执行回响逻辑
                    // 💡 这里我们硬编码 MaxLimit = 10，或者你可以从 req.body 读一个配置
                    // 假设用户希望总切片数维持在 10 个左右 (包括正常检索的 + 回响的)
                    let dynamicImpTags = ["important"]; // 默认兜底

                    if (
                        chatContext.strategy &&
                        chatContext.strategy.important &&
                        Array.isArray(chatContext.strategy.important.labels)
                    ) {
                        dynamicImpTags =
                            chatContext.strategy.important.labels.map((t) =>
                                t.toLowerCase(),
                            );
                        console.log(
                            `[Anima Echo] 🎯 动态重要标签: ${dynamicImpTags.join(", ")}`,
                        );
                    }

                    // 合并配置
                    const finalEchoConfig = {
                        ...(echoConfig || {}),
                        important_tags: dynamicImpTags,
                    };

                    const { echoItems, nextMemories, echoLogs } =
                        processEchoLogic(
                            finalChatResults,
                            globalTop50,
                            lastMemories, // ✅ 传入
                            finalEchoConfig,
                        );

                    // 🟢 将回响日志注入到 finalChatResults 的调试日志中
                    if (echoLogs && echoLogs.length > 0) {
                        const formattedEchoLogs = echoLogs.map((l) => {
                            const hasMeta =
                                l.meta && typeof l.meta === "object";
                            let displayTags = l.info;
                            if (
                                hasMeta &&
                                Array.isArray(l.meta.tags) &&
                                l.meta.tags.length > 0
                            ) {
                                displayTags = `${l.info} 🏷️[${l.meta.tags.join(", ")}]`;
                            }
                            return {
                                step: "Echo",
                                library: "Memory",
                                uniqueID: hasMeta ? l.meta.index : "-",
                                tags: displayTags,
                                score: hasMeta ? l.meta.score : 0,
                            };
                        });
                        collectedLogs.push(...formattedEchoLogs);
                    }

                    // 4. 合并结果
                    if (echoItems.length > 0) {
                        console.log(
                            `[Anima Echo] 🔗 成功回响插入 ${echoItems.length} 个旧记忆`,
                        );
                        finalChatResults = [...finalChatResults, ...echoItems];
                    }

                    saveSession(sessionId, {
                        lastUpdated: Date.now(),
                        memories: nextMemories,
                    });
                } catch (echoErr) {
                    console.error(
                        `[Anima Echo] ❌ 回响处理失败 (不影响主流程):`,
                        echoErr,
                    );
                }
            } else {
                console.log(
                    `[Anima Echo] ⚠️ 跳过回响 (无 SessionID 或 结果为空)`,
                );
            }

            if (finalChatResults.length > 0) {
                finalChatResults.sort((a, b) => {
                    const itemA = a.item.metadata;
                    const itemB = b.item.metadata;

                    // 1. Timestamp
                    const timeA = new Date(itemA.timestamp || 0).getTime();
                    const timeB = new Date(itemB.timestamp || 0).getTime();
                    if (timeA > 0 && timeB > 0 && timeA !== timeB) {
                        return timeA - timeB;
                    }

                    // 2. Index (Batch_Slice)
                    const parseId = (str) => {
                        const parts = (str || "0_0").split("_");
                        return {
                            batch: parseInt(parts[0] || 0),
                            slice: parseInt(parts[1] || 0),
                        };
                    };

                    const idA = parseId(itemA.index);
                    const idB = parseId(itemB.index);

                    if (idA.batch !== idB.batch) {
                        return idA.batch - idB.batch;
                    }
                    return idA.slice - idB.slice;
                });
            }

            // 6. 格式化输出函数
            const formatResults = (rawList) => {
                if (!rawList) return [];
                return rawList.map((r) => ({
                    text: r.item.metadata.text,
                    tags: r.item.metadata.tags,
                    score: r.score,
                    rerank_score: r._rerank_score,
                    timestamp: r.item.metadata.timestamp,
                    index: r.item.metadata.index,
                    chunk_index: r.item.metadata.chunk_index,
                    batch_id: r.item.metadata.batch_id,
                    source: r["_source_collection"] || "unknown",
                    doc_name: r.item.metadata.doc_name,
                    is_echo: r._is_echo || r.is_echo || false,
                    type: "vector",
                }));
            };
            const formattedVectorChat = formatResults(finalChatResults);
            const formattedVectorKb = formatResults(kbRaw);

            // ============================================================
            // 🧠 [新增] 后端核心：双轨去重与排序逻辑
            // ============================================================

            // 1. Chat 结果去重合并 (Vector + BM25)
            const mergeAndSortChat = (vecList, bm25List) => {
                const uniqueMap = new Map();
                const all = [...(vecList || []), ...(bm25List || [])];

                all.forEach((item) => {
                    // 使用 index (如 "1_2") 或 id 作为唯一键
                    const uniqueKey = item.index || item.id;
                    if (!uniqueMap.has(uniqueKey)) {
                        uniqueMap.set(uniqueKey, item);
                    }
                });

                const merged = Array.from(uniqueMap.values());

                // 严格按时间线排序
                merged.sort((a, b) => {
                    const timeA = new Date(a.timestamp || 0).getTime();
                    const timeB = new Date(b.timestamp || 0).getTime();
                    if (timeA !== timeB) return timeA - timeB;

                    const idxA = String(a.index || "0_0");
                    const idxB = String(b.index || "0_0");

                    const [batchA, sliceA] = idxA.split("_").map(Number);
                    const [batchB, sliceB] = idxB.split("_").map(Number);

                    if (isNaN(batchA) || isNaN(batchB)) {
                        return idxA.localeCompare(idxB, undefined, {
                            numeric: true,
                        });
                    }
                    if (batchA !== batchB) return batchA - batchB;
                    return (sliceA || 0) - (sliceB || 0);
                });
                return merged;
            };

            // 2. KB 结果去重合并 (Vector + BM25)
            const mergeAndSortKb = (vecList, bm25List) => {
                const uniqueMap = new Map();
                const all = [...(vecList || []), ...(bm25List || [])];

                all.forEach((item) => {
                    // 使用文档名+切片序号作为复合唯一键
                    const fallbackId =
                        (item.doc_name || "unknown") +
                        "_" +
                        (item.chunk_index || 0);
                    const uniqueKey = item.id || fallbackId;

                    if (!uniqueMap.has(uniqueKey)) {
                        uniqueMap.set(uniqueKey, item);
                    }
                });

                const merged = Array.from(uniqueMap.values());

                // 先按文档名称，再按切片序号排序
                merged.sort((a, b) => {
                    const docA = a.doc_name || "";
                    const docB = b.doc_name || "";
                    if (docA !== docB) return docA.localeCompare(docB);
                    return (a.chunk_index || 0) - (b.chunk_index || 0);
                });
                return merged;
            };

            const finalMergedChat = mergeAndSortChat(
                formattedVectorChat,
                bm25ChatResults,
            );
            const finalMergedKb = mergeAndSortKb(
                formattedVectorKb,
                bm25KbResults,
            );

            // 7. 返回扩充后的全量对象，满足前端所有 UI 模块的日志需求
            res.json({
                // [给 RAG / BM25 模块做独立日志用]
                vector_chat_results: formattedVectorChat,
                bm25_chat_results: bm25ChatResults,

                // [给 KB 模块做独立日志用]
                vector_kb_results: formattedVectorKb,
                bm25_kb_results: bm25KbResults,

                // [给底层步骤分析用]
                _debug_logs: collectedLogs,

                // [给拦截器直接注入 Prompt，以及主控台看最终结果用]
                merged_chat_results: finalMergedChat,
                merged_kb_results: finalMergedKb,
            });
        } catch (err) {
            console.error(err);
            res.status(500).json({
                success: false,
                message: err.message || "Unknown Query Error",
            });
        }
    });

    router.post("/export_collection", async (req, res) => {
        const { collectionId } = req.body;
        if (!collectionId) return res.status(400).send("Missing collectionId");

        const safeName = collectionId.replace(
            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
            "_",
        );
        const collectionPath = path.join(VECTOR_ROOT, safeName);

        if (!fs.existsSync(collectionPath)) {
            return res.status(404).send("Database not found");
        }

        try {
            const zip = new AdmZip();
            // 将整个文件夹添加到 zip
            zip.addLocalFolder(collectionPath);

            const buffer = zip.toBuffer();

            // 设置下载头
            res.set("Content-Type", "application/zip");

            // 🔥【核心修复】文件名编码
            // Node.js 禁止 Header 含中文。我们使用 encodeURIComponent 确保安全。
            // 前端 rag.js 会拦截 Blob 并重新命名，所以这里的文件名主要是为了协议合规。
            const encodedName = encodeURIComponent(safeName);

            res.set(
                "Content-Disposition",
                `attachment; filename="${encodedName}.zip"; filename*=UTF-8''${encodedName}.zip`,
            );

            res.set("Content-Length", buffer.length);
            res.send(buffer);

            console.log(`[Anima RAG] 📤 导出数据库成功: ${safeName}`);
        } catch (e) {
            console.error(`[Anima RAG] Export Error: ${e.message}`);
            // 只有当 Header 还没发出去时才发送 500，防止二次报错
            if (!res.headersSent) res.status(500).send(e.message);
        }
    });

    // ==========================================
    // 🟢 新增：检查数据库是否存在 (用于导入前的确认)
    // ==========================================
    router.post("/check_collection_exists", (req, res) => {
        const { collectionId } = req.body;
        const safeName = collectionId.replace(
            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
            "_",
        );
        const collectionPath = path.join(VECTOR_ROOT, safeName);

        res.json({ exists: fs.existsSync(collectionPath) });
    });

    // ==========================================
    // 🟢 新增：一键导入 (上传 ZIP)
    // ==========================================
    router.post("/import_collection", async (req, res) => {
        const { collectionId, zipData, force } = req.body; // zipData 是 base64 字符串

        if (!collectionId || !zipData)
            return res.status(400).send("Missing data");

        const safeName = collectionId.replace(
            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
            "_",
        );
        const collectionPath = path.join(VECTOR_ROOT, safeName);

        try {
            // 1. 检查是否存在
            if (fs.existsSync(collectionPath)) {
                if (!force) {
                    return res.json({ success: false, reason: "exists" });
                }
                // 强制覆盖：先删除旧文件夹
                fs.rmSync(collectionPath, { recursive: true, force: true });
            }

            // 2. 处理 Base64 并解压
            // 去掉 Data URI 前缀 (如 "data:application/zip;base64,")
            const base64Data = zipData.replace(/^data:.+;base64,/, "");
            const buffer = Buffer.from(base64Data, "base64");

            const zip = new AdmZip(buffer);
            zip.extractAllTo(collectionPath, true); // true = overwrite

            // 3. 强制清除可能存在的内存缓存，确保下次读取是新的
            if (activeIndexes.has(safeName)) activeIndexes.delete(safeName);

            console.log(`[Anima RAG] 📥 导入数据库成功: ${safeName}`);
            res.json({ success: true });
        } catch (e) {
            console.error(`[Anima RAG] Import Error: ${e.message}`);
            res.status(500).send(e.message);
        }
    });

    router.post("/merge", async (req, res) => {
        const { sourceIds, targetId } = req.body;

        if (!sourceIds || !Array.isArray(sourceIds) || sourceIds.length === 0) {
            return res.status(400).send("No source collections provided");
        }
        if (!targetId) {
            return res.status(400).send("Target collection ID is required");
        }

        const safeTargetName = targetId.replace(
            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
            "_",
        );

        console.log(`[Anima Merge] 🚀 开始合并任务 (深度读取模式)`);
        console.log(`   - 来源: ${sourceIds.join(", ")}`);
        console.log(`   - 目标: ${safeTargetName}`);

        try {
            await runInQueue(safeTargetName, async () => {
                // 1. 初始化目标库
                const targetIndex = await getIndex(safeTargetName, true);

                let successCount = 0;
                let failCount = 0;

                // 2. 遍历所有来源库
                for (const srcId of sourceIds) {
                    try {
                        // 获取源库的文件夹路径
                        const safeSrcName = srcId.replace(
                            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                            "_",
                        );
                        const srcFolderPath = path.join(
                            VECTOR_ROOT,
                            safeSrcName,
                        );

                        console.log(`[Anima Merge] 正在处理源库: ${srcId} ...`);

                        // 加载索引以获取 ID 列表和 Vector
                        const sourceIndex = await getIndex(srcId, false);
                        if (!sourceIndex) {
                            console.warn(
                                `[Anima Merge] ⚠️ 源库不存在，跳过: ${srcId}`,
                            );
                            continue;
                        }

                        const items = await sourceIndex.listItems();

                        // 3. 搬运条目 (这是核心修改部分)
                        for (const item of items) {
                            try {
                                // A. 构建源文件的物理路径
                                // vectra 通常用 item.id + ".json"
                                const fileName =
                                    item.metadataFile || `${item.id}.json`;
                                const filePath = path.join(
                                    srcFolderPath,
                                    fileName,
                                );

                                // B. 🔥 核心修复：必须从磁盘读取完整内容！
                                if (!fs.existsSync(filePath)) {
                                    console.warn(
                                        `[Anima Merge] ❌ 丢失物理文件: ${fileName}`,
                                    );
                                    failCount++;
                                    continue;
                                }

                                const fileContent = await fs.promises.readFile(
                                    filePath,
                                    "utf-8",
                                );
                                const fullData = JSON.parse(fileContent);

                                // C. 提取完整 Metadata (兼容数据结构)
                                // 有些版本数据直接在 root，有些在 metadata 字段下
                                const originalMetadata =
                                    fullData.metadata || fullData;

                                // D. 准备新的 Metadata
                                const newMetadata = {
                                    ...originalMetadata, // 这里面包含了 text, timestamp, tags 等所有数据
                                    _merge_source: srcId,
                                    _merged_at: Date.now(),
                                };

                                // E. 插入到目标库 (让 vectra 生成新 UUID)
                                await targetIndex.insertItem({
                                    vector: item.vector, // 向量可以直接从索引取，这个没问题
                                    metadata: newMetadata,
                                });

                                successCount++;
                            } catch (readErr) {
                                console.error(
                                    `[Anima Merge] 读取/写入单条失败 (${item.id}): ${readErr.message}`,
                                );
                                failCount++;
                            }
                        }
                    } catch (libErr) {
                        console.error(
                            `[Anima Merge] 处理源库 ${srcId} 异常: ${libErr.message}`,
                        );
                    }
                }

                console.log(
                    `[Anima Merge] ✅ 合并完成! 成功: ${successCount}, 失败: ${failCount}`,
                );

                res.json({
                    success: true,
                    targetId: safeTargetName,
                    stats: { success: successCount, failed: failCount },
                });
            });
        } catch (err) {
            console.error(`[Anima Merge] Critical Error: ${err.message}`);
            res.status(500).send(err.message);
        }
    });

    router.post("/rebuild_collection", async (req, res) => {
        const { collectionId, apiConfig } = req.body;

        if (!collectionId) return res.status(400).send("Missing collectionId");
        // 校验 API 配置
        if (!apiConfig || !apiConfig.key)
            return res.status(400).send("Missing API Config");

        const safeName = collectionId.replace(
            /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
            "_",
        );

        console.log(`[Anima Rebuild] 🚀 开始重建库: ${safeName}`);

        try {
            await runInQueue(safeName, async () => {
                const targetIndex = await getIndex(safeName, false); // 必须存在
                if (!targetIndex) {
                    throw new Error("Collection not found");
                }

                const items = await targetIndex.listItems();
                const folderPath = path.join(VECTOR_ROOT, safeName);

                let successCount = 0;
                let failCount = 0;

                // 遍历所有条目
                for (const item of items) {
                    try {
                        // 1. 读取物理文件获取文本
                        const fileName = item.metadataFile || `${item.id}.json`;
                        const filePath = path.join(folderPath, fileName);

                        if (!fs.existsSync(filePath)) {
                            console.warn(
                                `[Anima Rebuild] ❌ 文件丢失: ${fileName}`,
                            );
                            failCount++;
                            continue;
                        }

                        const fileContent = await fs.promises.readFile(
                            filePath,
                            "utf-8",
                        );
                        const fullData = JSON.parse(fileContent);
                        // 兼容元数据位置
                        const meta = fullData.metadata || fullData;
                        const text = meta.text;

                        if (!text) {
                            console.warn(
                                `[Anima Rebuild] ⚠️ 条目无文本，跳过: ${item.id}`,
                            );
                            failCount++;
                            continue;
                        }

                        // 2. 重新向量化 (调用 OpenAI/DeepSeek)
                        // 注意：这里是串行的，速度较慢，但安全
                        const newVector = await getEmbedding(text, apiConfig);

                        // 3. 更新数据库
                        // Vectra 没有原地的 update，我们需要：先删 -> 后加
                        // 为了保留原来的 ID 引用（如果有外部依赖），理想情况是保留 ID。
                        // 但 RAG 系统通常只依赖内容，生成新 ID 也是安全的。
                        // 为了稳妥，我们采用“删除旧条目 -> 插入新条目”

                        // A. 删除旧的
                        await targetIndex.deleteItem(item.id);
                        // 同时删除旧物理文件（因为 insertItem 会生成新的）
                        if (fs.existsSync(filePath)) fs.unlinkSync(filePath);

                        // B. 插入新的 (携带旧的 metadata)
                        await targetIndex.insertItem({
                            vector: newVector,
                            metadata: meta, // 包含 timestamp, index, batch_id, tags 等
                        });

                        successCount++;

                        // 简单的后端日志进度
                        if (successCount % 5 === 0)
                            console.log(
                                `[Anima Rebuild] ${safeName}: ${successCount}/${items.length}`,
                            );
                    } catch (err) {
                        console.error(
                            `[Anima Rebuild] 单条失败 (${item.id}): ${err.message}`,
                        );
                        failCount++;
                    }
                }

                console.log(
                    `[Anima Rebuild] ✅ 库 ${safeName} 重建完毕. 成功: ${successCount}, 失败: ${failCount}`,
                );

                res.json({
                    success: true,
                    collectionId: safeName,
                    stats: { success: successCount, failed: failCount },
                });
            });
        } catch (err) {
            console.error(`[Anima Rebuild] Error: ${err.message}`);
            // 发送 500 会导致前端 catch，包含错误信息
            res.status(500).send(err.message);
        }
    });

    // API: 物理删除整个向量库文件夹 (慎用)
    router.post("/delete_collection", async (req, res) => {
        const { collectionId } = req.body;

        // 安全检查：不允许删除空名或根目录
        if (
            !collectionId ||
            collectionId.trim() === "" ||
            collectionId.includes("..") ||
            collectionId.includes("/") ||
            collectionId.includes("\\")
        ) {
            return res.status(400).send("Invalid or unsafe collectionId");
        }

        try {
            // 1. 先从内存缓存中移除
            if (activeIndexes.has(collectionId)) {
                activeIndexes.delete(collectionId);
            }
            if (writeQueues.has(collectionId)) {
                writeQueues.delete(collectionId);
            }

            // 2. 构建路径
            const safeName = collectionId.replace(
                /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                "_",
            );
            const collectionPath = path.join(VECTOR_ROOT, safeName);

            // 3. 检查是否存在
            if (!fs.existsSync(collectionPath)) {
                return res.json({
                    success: true,
                    message: "Folder did not exist",
                });
            }

            // 4. 物理删除 (递归)
            // Node.js 14.14+ 支持 { recursive: true }
            fs.rmSync(collectionPath, { recursive: true, force: true });

            // 🌟 移除了强制删除 BM25 库的代码，实现两库分离独立删除

            console.log(`[Anima RAG] 🗑️ 向量数据库已物理删除: ${collectionId}`);
            res.json({ success: true });
        } catch (err) {
            console.error(
                `[Anima RAG] Delete Collection Error: ${err.message}`,
            );
            res.status(500).send(err.message);
        }
    });

    router.post("/delete_batch", async (req, res) => {
        const { collectionId, batch_id } = req.body;

        // 使用队列包装，确保安全
        await runInQueue(collectionId, async () => {
            if (!collectionId || batch_id === undefined) {
                // 注意：这里是在 async 回调里，不能直接 return res
                // 必须抛出错误让 runInQueue 的 catch 捕获，或者在这里发送响应
                res.status(400).send("Missing collectionId or batch_id");
                return;
            }

            const targetIndex = await getIndex(collectionId);

            // 1. 强制重新加载，确保拿到磁盘最新状态
            // (LocalIndex 有时会缓存旧数据，虽然我们加了 activeIndexes，但为了保险起见，listItems 是安全的)
            const allItems = await targetIndex.listItems();

            // 2. 筛选目标 (严格字符串比对)
            // 注意：一定要做 String 转换，防止 json 里是数字而参数是字符串导致漏选
            const targets = allItems.filter(
                (item) =>
                    item.metadata &&
                    String(item.metadata.batch_id) === String(batch_id),
            );

            if (targets.length === 0) {
                console.log(
                    `[Anima RAG] Batch ${batch_id} 无旧数据，无需删除。`,
                );
                res.json({ success: true, count: 0 });
                return;
            }

            console.log(
                `[Anima RAG] 🔍 发现 Batch ${batch_id} 待删除条目: ${targets.length} 个`,
            );

            // =========================================================
            // 🔥 核心修复：预先构建“死刑名单” (Deletion Plan)
            // 防止在删除过程中 item 对象属性丢失或索引状态改变
            // =========================================================
            const deletionPlan = targets.map((item) => {
                // 构建绝对路径
                // 假设 collectionId 本身就是文件夹名（经过 safeName 处理）
                const safeName = collectionId.replace(
                    /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                    "_",
                );
                const collectionPath = path.join(VECTOR_ROOT, safeName);

                return {
                    id: item.id,
                    // 确保拿到 metadataFile，如果不存在则为 null
                    filePath: item.metadataFile
                        ? path.join(collectionPath, item.metadataFile)
                        : null,
                };
            });

            // =========================================================
            // 3. 执行处决 (Execute Deletion)
            // =========================================================
            let deletedCount = 0;
            let physicalDeleteCount = 0;

            for (const plan of deletionPlan) {
                try {
                    // A. 逻辑删除 (从 index.json 移除)
                    await targetIndex.deleteItem(plan.id);
                    deletedCount++;

                    // B. 物理删除 (从磁盘移除 .json)
                    if (plan.filePath) {
                        if (fs.existsSync(plan.filePath)) {
                            fs.unlinkSync(plan.filePath);
                            physicalDeleteCount++;
                            // console.log(`[Anima] 🗑️ 文件已删: ${path.basename(plan.filePath)}`);
                        } else {
                            // 文件不存在可能是已经被删了，或者路径不对，打印个警告以便调试
                            console.warn(
                                `[Anima] ⚠️ 文件未找到 (跳过): ${plan.filePath}`,
                            );
                        }
                    }
                } catch (err) {
                    console.error(
                        `[Anima] 删除单条失败 (ID: ${plan.id}): ${err.message}`,
                    );
                }
            }

            const idsToDelete = deletionPlan.map((p) => p.id);
            if (idsToDelete.length > 0) {
                // 加 catch 防止由于 BM25 没数据报错影响主流程
                await bm25Engine
                    .deleteDocuments(collectionId, idsToDelete)
                    .catch((e) => console.error(e));
            }

            console.log(
                `[Anima RAG] 🧹 Batch ${batch_id} 清理完毕: 索引删除了 ${deletedCount} 个, 物理文件删除了 ${physicalDeleteCount} 个`,
            );

            // 4. 响应前端
            res.json({
                success: true,
                count: deletedCount,
                physicalCount: physicalDeleteCount,
            });
        });
    });

    router.post("/delete", async (req, res) => {
        const { collectionId, index } = req.body;

        await runInQueue(collectionId, async () => {
            if (!collectionId || index === undefined) {
                res.status(400).send("Missing collectionId or index");
                return;
            }

            console.log(
                `[Anima RAG] 收到删除请求: Collection=${collectionId}, Index=${index}`,
            );

            const targetIndex = await getIndex(collectionId);
            const allItems = await targetIndex.listItems();

            // 1. 筛选目标
            const targets = allItems.filter(
                (item) =>
                    item.metadata &&
                    String(item.metadata.index) === String(index),
            );

            if (targets.length === 0) {
                console.log(
                    `[Anima RAG] 未找到 Index ${index} 的记录，跳过删除。`,
                );
                res.json({ success: true, count: 0 });
                return;
            }

            // 2. 构建“死刑名单” (Deletion Plan)
            const safeName = collectionId.replace(
                /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                "_",
            );
            const collectionPath = path.join(VECTOR_ROOT, safeName);

            const deletionPlan = targets.map((item) => ({
                id: item.id,
                filePath: item.metadataFile
                    ? path.join(collectionPath, item.metadataFile)
                    : null,
            }));

            // 3. 执行删除
            let deletedCount = 0;
            let physicalDeleteCount = 0;

            for (const plan of deletionPlan) {
                try {
                    // A. 逻辑删除
                    await targetIndex.deleteItem(plan.id);
                    deletedCount++;

                    // B. 物理删除
                    if (plan.filePath) {
                        if (fs.existsSync(plan.filePath)) {
                            fs.unlinkSync(plan.filePath);
                            physicalDeleteCount++;
                            // console.log(`[Anima] 🗑️ 单条物理文件已删: ${path.basename(plan.filePath)}`);
                        }
                    }
                } catch (e) {
                    console.warn(`[Anima RAG] 单条删除异常: ${e.message}`);
                }
            }

            const idsToDelete = deletionPlan.map((p) => p.id);
            if (idsToDelete.length > 0) {
                await bm25Engine
                    .deleteDocuments(collectionId, idsToDelete)
                    .catch((e) => console.error(e));
            }

            console.log(
                `[Anima RAG] ✅ Index ${index} 删除完成: 索引-${deletedCount}, 文件-${physicalDeleteCount}`,
            );
            res.json({
                success: true,
                count: deletedCount,
                physicalCount: physicalDeleteCount,
            });
        });
    });

    router.post("/proxy/forward", async (req, res) => {
        const { targetUrl, method, headers, body, isStream } = req.body;
        if (!targetUrl)
            return res.status(400).json({ error: "Missing targetUrl" });

        try {
            console.log(
                `[Anima Proxy] 🌐 转发 -> ${method || "GET"} ${targetUrl}`,
            );
            const fetchOptions = {
                method: method || "GET",
                headers: headers || {},
            };
            if (body)
                fetchOptions.body =
                    typeof body === "string" ? body : JSON.stringify(body);

            // =========================================================
            // ✨ 核心修改：根据 ST 的配置智能判断是否挂载代理
            // =========================================================
            let useProxy = false;

            if (stProxyConfig.enabled && stProxyConfig.url) {
                useProxy = true;
                // 检查是否在 bypass (直连) 豁免名单中
                for (const bypassUrl of stProxyConfig.bypass) {
                    if (targetUrl.includes(bypassUrl)) {
                        useProxy = false;
                        break;
                    }
                }
            }

            if (useProxy) {
                // 挂载 ST 配置的动态代理
                fetchOptions.dispatcher = new ProxyAgent(stProxyConfig.url);
                console.log(
                    `[Anima Proxy] 🛡️ 正在使用代理转发: ${stProxyConfig.url}`,
                );
            } else {
                // ⚡ 不配置 dispatcher，原生走直连
                console.log(`[Anima Proxy] ⚡ 正在使用直连转发 (无代理)`);
            }

            const response = await fetch(targetUrl, fetchOptions);

            if (!response.ok) {
                const errText = await response.text();
                return res.status(response.status).send(errText);
            }

            if (isStream) {
                res.setHeader("Content-Type", "text/event-stream");
                res.setHeader("Cache-Control", "no-cache");
                res.setHeader("Connection", "keep-alive");
                if (response.body.pipe) {
                    response.body.pipe(res);
                } else {
                    const reader = response.body.getReader();
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        res.write(Buffer.from(value));
                    }
                    res.end();
                }
            } else {
                const data = await response.json();
                res.json(data);
            }
        } catch (error) {
            console.error(`[Anima Proxy] 崩溃:`, error);
            res.status(500).json({ error: error.message });
        }
    });

    console.log("[Anima RAG] 后端服务已启动 (支持多聊天隔离)");
}

module.exports = {
    init,
    exit: async () => {},
    info: {
        id: "anima-rag",
        name: "Anima Project RAG",
        description: "Anima RAG Backend with Batch/Slice support",
    },
};
