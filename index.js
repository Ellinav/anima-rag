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

const path = require("path");
const fs = require("fs");
const { LocalIndex } = require("vectra");
const AdmZip = require("adm-zip");

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

    const maxTotalLimit = config.max_count || 10;
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
                `[Anima Echo] ♻️ [刷新] Index ${indexStr} (自然命中) | Life重置: ${memory.maxLife}`,
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
                        `[Anima Echo] 🔗 [回响成功] Index ${indexStr} | 剩余Life: ${newLife}`,
                        metaData, // 这里的 metaData 已经是最新分数了，所以你的浏览器控制台是对的
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
                        `[Anima Echo] ⏳ [排队等待] Index ${indexStr} (无卡槽) | 剩余Life: ${newLife}`,
                        metaData,
                    );
                }
            } else {
                // Life 本来就是 0 (且没被复活)，那就真的死了
                log(
                    `[Anima Echo] 💀 [记忆枯竭] Index ${indexStr} (Life耗尽 -> 删除)`,
                    metaData,
                );
            }
        } else {
            // [B. 离题] - 直接移除，不进入 nextMemories
            log(
                `[Anima Echo] 💨 [遗忘] Index ${indexStr} (脱离相关性范围)`,
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

                // 打印后端加权日志
                const indexStr = res.item.metadata?.index || "unknown";
                console.log(
                    `[Anima 加权] ⬆️ Index ${indexStr} | 分数: ${oldScore.toFixed(4)} -> ${res.score.toFixed(4)} (+${recentWeight})`,
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

// 🔥 新增：动态策略执行器 (2-1-2-N*M 完整逻辑 - 最终修复版)
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
    console.log(
        `[Anima RAG] 🚀 执行策略 | 步骤数: ${steps.length} | 排除ID: ${ignoreIds.length} 个`,
    );

    // =========================================================
    // 🔥 核心修改：动态构建“功能性标签池” (Functional Tags Pool)
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

    console.log(
        `[Anima Debug] 动态识别的功能性标签:`,
        Array.from(functionalTagsSet),
    );

    // =========================================================
    // 🛠️ 辅助函数：构建合并过滤器
    // =========================================================
    const buildFilter = (stepFilter = {}) => {
        if (!ignoreIds || ignoreIds.length === 0) return stepFilter;
        return {
            ...stepFilter,
            index: { $nin: ignoreIds },
        };
    };

    // =========================================================
    // 循环执行步骤
    // =========================================================

    // 记录 Step 1 发现的风格 (用于 Step 6 排除)
    let detectedVibeTag = null;
    let detectedImportantLabels = [];

    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        if (step.count <= 0) continue;

        // 1. 定义硬编码系数 (Hardcoded Coefficients)
        let stepCoeff = 1.0;
        let useGlobalMultiplier = true;

        switch (step.type) {
            case "base":
                stepCoeff = 5;
                break;
            case "important":
                // 重要检索：极易重复，需要深挖 (例如: 全局2.0 * 2.5 = 5倍候选)
                stepCoeff = 2;
                break;
            case "diversity":
                // 丰富度：需要跳过前几步已选的 ID，适当增加冗余
                stepCoeff = 1.5;
                break;
            default:
                // 状态、生理、节日等：适度冗余
                stepCoeff = 1.5;
                break;
        }

        // 2. 计算最终倍率
        const finalMultiplier = useGlobalMultiplier
            ? multiplier * stepCoeff
            : stepCoeff;

        // 3. 计算 candidateK (保留 Math.max(..., 2) 作为最小安全值)
        const candidateK = Math.max(Math.ceil(step.count * finalMultiplier), 2);

        // [调试日志] 方便你观察实际用了多少倍率
        console.log(
            `[Step ${i + 1} - ${step.type}] Count: ${step.count} | Multiplier: ${finalMultiplier.toFixed(1)}x | Candidates Per DB: ${candidateK}`,
        );

        let candidates = [];
        switch (step.type) {
            case "base":
                candidates = await queryMultiIndices(
                    indices,
                    vector,
                    candidateK, // 这里是动态计算出来的 (e.g., 20)
                    buildFilter({}),
                    "Chat",
                    recentWeight,
                    currentSessionId,
                );

                // 🕵️ 智能捕获 Vibe Tag (排除法)
                if (!detectedVibeTag && candidates.length > 0) {
                    const topItem = candidates[0].item.metadata;
                    const topTags = topItem.tags || [];

                    // 逻辑：遍历 Top 1 切片的所有标签
                    // 只要这个标签不在 functionalTagsSet 里，它就是 Vibe
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

                // 2. 截取全局 Top 50，存入我们定义的变量中
                if (echoCandidatePool.length === 0) {
                    echoCandidatePool = candidates.slice(0, 50);
                    console.log(
                        `[Anima Strategy] 🌊 已捕获 Base 全局池用于回响 (Top ${echoCandidatePool.length})`,
                    );
                }
                // 注意：这里不需要手动 slice candidates 给后续逻辑
                // 因为原本的逻辑下面有 `for (const res of candidates) { if (addedInStep >= limit) break; ... }`
                // 它会自动只取 step.count (比如 5 个) 放入 finalResults
                break;

            case "important":
                if (step.labels && step.labels.length > 0) {
                    detectedImportantLabels = step.labels; // 记录用于后续排除

                    const impPromises = step.labels.map((label) =>
                        queryMultiIndices(
                            indices,
                            vector,
                            candidateK,
                            buildFilter({ tags: { $in: [label] } }),
                            "Chat",
                            recentWeight,
                            currentSessionId,
                        ),
                    );
                    const impResults = await Promise.all(impPromises);

                    // 强制均衡：确保每个标签都贡献 step.count 个结果
                    const balancedResults = impResults.map((list) => {
                        list.sort((a, b) => b.score - a.score);
                        return list; // ✅ 改为直接返回完整列表
                    });

                    candidates = balancedResults.flat();
                    console.log(
                        `   [Important] 分路检索: 触发 ${step.labels.join(", ")}`,
                    );
                }
                break;

            case "status":
                // Step 3: 状态检索 (Fan-out)
                if (step.labels && step.labels.length > 0) {
                    const statusPromises = step.labels.map((label) =>
                        queryMultiIndices(
                            indices,
                            vector,
                            candidateK,
                            buildFilter({ tags: { $in: [label] } }),
                            "Chat",
                            recentWeight,
                            currentSessionId,
                        ),
                    );
                    const statusResults = await Promise.all(statusPromises);

                    // 🔥 [核心修正] 强制均衡策略
                    // 如果 step.count 是 1，我们要确保每个标签都贡献 1 条，
                    // 而不是把所有结果混在一起按分数排序（那样可能会导致高分标签挤掉低分标签）。
                    // 所以我们在合并前，先对每个结果集进行截断。
                    const balancedResults = statusResults.map((list) => {
                        list.sort((a, b) => b.score - a.score);
                        // ❌ 删除这行: return list.slice(0, step.count);
                        return list; // ✅
                    });

                    candidates = balancedResults.flat();
                    console.log(
                        `   [Status] 分支检索: 触发 ${step.labels.join(", ")} (均衡模式)`,
                    );
                }
                break;

            case "period":
                // Step 4: 生理检索
                if (step.labels && step.labels.length > 0) {
                    candidates = await queryMultiIndices(
                        indices,
                        vector,
                        candidateK,
                        buildFilter({ tags: { $in: step.labels } }), // <--- 修改点
                        "Chat",
                        recentWeight,
                        currentSessionId,
                    );
                }
                break;

            case "special":
                // Step 5: 节日检索 (修改版：支持多节日分路)
                // 前端现在会传 labels: ["Birthday", "Christmas"]
                if (step.labels && step.labels.length > 0) {
                    const specialPromises = step.labels.map((label) =>
                        queryMultiIndices(
                            indices,
                            vector,
                            candidateK,
                            buildFilter({ tags: { $in: [label] } }),
                            "Chat",
                            recentWeight,
                            currentSessionId,
                        ),
                    );
                    const specialResults = await Promise.all(specialPromises);

                    const balancedResults = specialResults.map((list) => {
                        list.sort((a, b) => b.score - a.score);
                        return list; // ✅
                    });

                    candidates = balancedResults.flat();
                    console.log(
                        `   [Special] 节日触发: ${step.labels.join(", ")}`,
                    );
                }
                // 兼容旧逻辑 (防止前端没更新导致报错)
                else if (step.target_tag) {
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
                // Step 6: 丰富度检索
                const excludeTags = [...detectedImportantLabels];
                if (detectedVibeTag && !excludeTags.includes(detectedVibeTag)) {
                    excludeTags.push(detectedVibeTag);
                }

                if (excludeTags.length > 0) {
                    candidates = await queryMultiIndices(
                        indices,
                        vector,
                        candidateK,
                        buildFilter({ tags: { $nin: excludeTags } }), // <--- 修改点
                        "Chat",
                        recentWeight,
                        currentSessionId,
                    );
                } else {
                    candidates = await queryMultiIndices(
                        indices,
                        vector,
                        candidateK,
                        buildFilter({}),
                        "Chat",
                        recentWeight,
                        currentSessionId,
                    );
                }
                break;
        }

        if (candidates.length > 0) {
            candidates.forEach((c) => {
                // 记录每一条候选项的来源信息
                debugLogs.push({
                    step: `Step ${i + 1}: ${step.type.toUpperCase()}`,
                    library: c._source_collection,
                    uniqueID: c.item.metadata.index,
                    tags: (c.item.metadata.tags || []).join(", "),
                    // 🟢 [修改这里] 如果被加权过，在浏览器日志里加个 ⬆️ 箭头提示
                    score: c._is_weighted
                        ? `${c.score.toFixed(4)} (⬆️+${recentWeight})`
                        : c.score.toFixed(4),
                });
            });
        }

        // === 聚合结果 (去重 & 阈值) ===
        let addedInStep = 0;
        // 如果是 Status Fan-out，candidates 可能很多，先排序
        candidates.sort((a, b) => b.score - a.score);

        // 计算这一步允许的最大数量 (Status步骤如果有多路，允许总量增加)
        const limit =
            step.type === "status" && step.labels
                ? step.count * step.labels.length
                : step.count;

        for (const res of candidates) {
            if (addedInStep >= limit) break;
            if (usedIds.has(res.item.id)) continue;

            // 阈值检查 (特殊步骤可适当放宽，此处暂统一标准)
            if (res.score < globalMinScore) {
                // 可选：给予 Important/Special 0.1 的豁免权
                if (
                    ["important", "status", "period", "special"].includes(
                        step.type,
                    )
                ) {
                    if (res.score < Math.max(0, globalMinScore - 0.2)) continue;
                } else {
                    continue;
                }
            }

            finalResults.push(res);
            usedIds.add(res.item.id);
            addedInStep++;
        }
    }

    // Final Sort: 按时间倒序 (旧 -> 新) 以符合阅读习惯，或者按相关性
    // 这里保持：Narrative Time (旧->新) -> Index
    finalResults.sort((a, b) => {
        const itemA = a.item.metadata;
        const itemB = b.item.metadata;

        // 1. 优先尝试按时间排序 (如果有有效时间)
        const timeA = new Date(itemA.timestamp || 0).getTime();
        const timeB = new Date(itemB.timestamp || 0).getTime();
        // 只有当时间有显著差异（且不为0）时才生效，否则视为时间相同，走 ID 排序
        if (timeA > 0 && timeB > 0 && timeA !== timeB) {
            return timeA - timeB;
        }

        // 2. 核心修正：按 Batch_Slice 完整排序
        // 解析 ID，例如 "5_12" -> batch:5, slice:12
        const parseId = (str) => {
            const parts = (str || "0_0").split("_");
            return {
                batch: parseInt(parts[0] || 0),
                slice: parseInt(parts[1] || 0),
            };
        };

        const idA = parseId(itemA.index);
        const idB = parseId(itemB.index);

        // 先比 Batch (批次)
        if (idA.batch !== idB.batch) {
            return idA.batch - idB.batch;
        }
        // 再比 Slice (切片)
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
                        for (const plan of deletionPlan) {
                            try {
                                await targetIndex.deleteItem(plan.id); // 删索引
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
                        // ✅ 这里使用处理过的 safeBatchId，而不是原始的 parseInt(batch_id)
                        batch_id: safeBatchId,
                    },
                });

                console.log(
                    `[Anima RAG] ✅ 写入成功 | Batch: ${safeBatchId} | Index: ${index}`,
                );
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
        const { fileName, fileContent, settings, apiConfig } = req.body;

        if (!fileName || !fileContent)
            return res.status(400).send("Missing file data");

        // 1. 构造 Collection ID (格式: kb_文件名)
        // 去除扩展名并进行安全处理
        const safeName = fileName
            .replace(/\.[^/.]+$/, "")
            .replace(/[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g, "_");
        const collectionId = `kb_${safeName}`;

        console.log(`[Anima KB] 📚 正在处理知识库: ${collectionId}`);

        try {
            await runInQueue(collectionId, async () => {
                // 2. 获取或创建 Index (允许创建)
                const targetIndex = await getIndex(collectionId, true);

                // 3. 覆盖逻辑：如果已存在，先清空
                // Vectra 没有直接 truncate，我们简单地遍历删除或直接删文件重建
                // 这里为了稳妥，采用“删文件重建”逻辑（复用之前的 delete_collection 逻辑的一部分）
                // 简单做法：如果 items > 0，则物理删除文件夹后重新 new LocalIndex
                const stats = await targetIndex.listItems();
                if (stats.length > 0) {
                    console.log(`[Anima KB] 发现旧数据，正在重建库...`);
                    const folderPath = path.join(VECTOR_ROOT, collectionId);
                    if (fs.existsSync(folderPath)) {
                        fs.rmSync(folderPath, { recursive: true, force: true });
                        // 必须移除内存缓存，否则 LocalIndex 还是指向旧的句柄
                        activeIndexes.delete(collectionId);
                    }
                    // 重新获取新实例
                    // 注意：这里需要递归调用自己或者简单地重新 getIndex
                    // 由于上面删了 activeIndexes，再次 getIndex 会重新创建
                }

                // 重新获取干净的 index
                const cleanIndex = await getIndex(collectionId, true);

                // 4. 切片
                const chunks = chunkText(fileContent, {
                    delimiter: settings.delimiter,
                    chunkSize: settings.chunk_size,
                });

                console.log(
                    `[Anima KB] 切片完成，共 ${chunks.length} 个片段。开始向量化...`,
                );

                // 5. 批量向量化 (串行，防止 API 速率限制)
                for (let i = 0; i < chunks.length; i++) {
                    const chunkText = chunks[i];
                    try {
                        const vector = await getEmbedding(chunkText, apiConfig);
                        await cleanIndex.insertItem({
                            vector: vector,
                            metadata: {
                                text: chunkText,
                                doc_name: fileName,
                                source_type: "knowledge", // 标记类型
                                chunk_index: i,
                                timestamp: Date.now(),
                            },
                        });
                        // 简单的进度日志
                        if ((i + 1) % 10 === 0)
                            console.log(
                                `[Anima KB] 进度: ${i + 1}/${chunks.length}`,
                            );
                    } catch (err) {
                        console.error(
                            `[Anima KB] 片段 ${i} 向量化失败:`,
                            err.message,
                        );
                    }
                }
            });

            res.json({ success: true, collectionId: collectionId, count: 0 }); // count 暂不返回准确值以免复杂
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

    router.post("/view_collection", async (req, res) => {
        const { collectionId } = req.body;
        if (!collectionId) return res.status(400).send("Missing collectionId");

        try {
            // 1. 获取索引实例
            const targetIndex = await getIndex(collectionId, false);

            if (!targetIndex) {
                return res.status(404).json({ error: "Database not found" });
            }

            // 2. 获取所有条目索引
            const indexItems = await targetIndex.listItems();

            // 3. 准备路径工具
            const safeName = collectionId.replace(
                /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
                "_",
            );
            const collectionPath = path.join(VECTOR_ROOT, safeName);

            // 4. 🟢 核心修复：直接读取磁盘文件
            const formattedItems = await Promise.all(
                indexItems.map(async (entry) => {
                    try {
                        // 优先使用 entry.metadataFile (vectra 可能会提供)，如果没有则尝试 id.json
                        // 如果你的 vectra 版本不提供 metadataFile，通常文件名就是 id.json
                        const fileName =
                            entry.metadataFile || `${entry.id}.json`;
                        const filePath = path.join(collectionPath, fileName);

                        if (!fs.existsSync(filePath)) {
                            console.warn(
                                `[Anima RAG] ⚠️ 文件丢失: ${fileName}`,
                            );
                            return null;
                        }

                        const fileContent = await fs.promises.readFile(
                            filePath,
                            "utf-8",
                        );
                        const fullData = JSON.parse(fileContent);

                        // 兼容性处理：数据可能在 root，也可能在 metadata 字段下
                        // 根据你提供的 UUID.json 内容，数据似乎平铺在 root 或者 metadata 里
                        // 我们做一个合并策略以防万一
                        const meta = fullData.metadata || fullData;

                        return {
                            id: entry.id,
                            text: meta.text || "",
                            metadata: {
                                chunk_index: meta.chunk_index, // 这样一定能取到
                                doc_name: meta.doc_name,
                                timestamp: meta.timestamp,
                            },
                        };
                    } catch (e) {
                        console.warn(
                            `[Anima RAG] 读取切片失败 (${entry.id}): ${e.message}`,
                        );
                        return null;
                    }
                }),
            );

            // 过滤掉读取失败的
            const validItems = formattedItems.filter((i) => i !== null);

            // 🟢 调试日志：打印第一条数据，看看长什么样
            if (validItems.length > 0) {
                console.log(
                    "[Anima Debug] 第一条数据预览:",
                    JSON.stringify(validItems[0].metadata),
                );
            }

            console.log(
                `[Anima RAG] 👀 查看库: ${collectionId} | 磁盘读取: ${validItems.length}`,
            );

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
            // 1. 获取基础参数 (这里解构后，sessionId 就已经存在了)
            const {
                searchText,
                apiConfig,
                ignore_ids,
                echoConfig,
                sessionId,
                is_swipe,
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
            if (!searchText)
                return res.json({ chat_results: [], kb_results: [] });
            const vector = await getEmbedding(searchText, apiConfig);

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

            // --- 任务 A: 聊天记录检索 ---
            const chatTask = async () => {
                const targetIds = Array.isArray(chatContext.ids)
                    ? chatContext.ids.filter((id) => id)
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

                const safeIgnoreIds = Array.isArray(ignore_ids)
                    ? ignore_ids
                    : [];
                const strat = chatContext.strategy;

                if (strat && strat.enabled) {
                    return await performDynamicStrategy(
                        uniqueIndices,
                        vector,
                        strat,
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
                    timestamp: r.item.metadata.timestamp,
                    index: r.item.metadata.index,
                    batch_id: r.item.metadata.batch_id,
                    source: r["_source_collection"] || "unknown",
                    doc_name: r.item.metadata.doc_name,
                    is_echo: r._is_echo || r.is_echo || false,
                }));
            };
            const executionLogs = finalChatResults["_debug_logs"] || [];

            // 7. 返回合并对象
            res.json({
                chat_results: formatResults(finalChatResults),
                kb_results: formatResults(kbRaw),
                _debug_logs: collectedLogs,
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

            console.log(`[Anima RAG] 🗑️ 整个数据库已物理删除: ${collectionId}`);
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
