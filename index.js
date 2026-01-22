const path = require("path");
const fs = require("fs");
const { LocalIndex } = require("vectra");

const VECTOR_ROOT = path.join(__dirname, "vectors");
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

// 辅助：获取向量
async function getEmbedding(text, config) {
    if (!config || !config.key) throw new Error("API Key missing");
    try {
        const fetchUrl = `${config.url.replace(/\/+$/, "")}/embeddings`;
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
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        return data.data[0].embedding;
    } catch (error) {
        console.error("[Anima RAG] Embedding Failed:", error);
        throw error;
    }
}

/**
 * 多库聚合检索辅助函数
 * @param {any[]} indices - 向量库实例数组
 * @param {number[]} vector - 查询向量
 * @param {number} k - 需要获取的数量
 * @param {object} [filter=null] - (可选) 过滤条件，默认为 null
 */
async function queryMultiIndices(indices, vector, k, filter = null) {
    // 1. 并行查询所有 Index
    const promises = indices.map((idx) =>
        idx
            .queryItems(vector, k, filter) // filter 会被正确透传给 vectra
            .catch((e) => {
                // console.warn(`[Anima RAG] 单库查询失败 (忽略): ${e.message}`);
                return [];
            }),
    );

    const resultsArrays = await Promise.all(promises);

    // 2. 拍平结果数组
    let allResults = resultsArrays.flat();

    // 3. 全局按分数降序排序 (High -> Low)
    allResults.sort((a, b) => b.score - a.score);

    // 4. 截取全局 Top K
    return allResults.slice(0, k);
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
async function getIndex(collectionId) {
    if (!collectionId) throw new Error("Collection ID is required");

    const safeName = collectionId.replace(
        /[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g,
        "_",
    );

    if (activeIndexes.has(safeName)) return activeIndexes.get(safeName);
    if (loadingPromises.has(safeName)) return loadingPromises.get(safeName);

    const loadTask = (async () => {
        const collectionPath = path.join(VECTOR_ROOT, safeName);
        console.log(`[Anima Debug] 📂 Loading Index: ${safeName}`);

        if (!fs.existsSync(collectionPath))
            fs.mkdirSync(collectionPath, { recursive: true });

        const indexInstance = new LocalIndex(collectionPath);
        if (!(await indexInstance.isIndexCreated())) {
            await indexInstance.createIndex({
                version: 1,
                metadata_config: { indexed: ["tags", "index", "batch_id"] },
            });
        }

        // 强制预热
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
        activeIndexes.set(safeName, instance);
        return instance;
    } finally {
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

async function queryMultiIndices(indices, vector, k, filter = null) {
    console.log(`[Anima Debug] 🚀 并行检索 ${indices.length} 个库...`);

    const promises = indices.map(
        (idx) => queryIndexSafe(idx, vector, k, filter),
        // 注意：这里不需要再 catch 了，因为 queryIndexSafe 内部已经 catch 并返回 []
        // 这样写更干净
    );

    const resultsArrays = await Promise.all(promises);

    // 拍平结果
    let allResults = resultsArrays.flat();
    console.log(
        `[Anima Debug] 📊 聚合所有库结果，共 ${allResults.length} 条 (排序前)`,
    );

    // 排序
    allResults.sort((a, b) => b.score - a.score);

    // 截取
    return allResults.slice(0, k);
}

// 🔥 性能与精度拉满：分布式精准检索策略 (2-1-2-1)
async function perform2121Strategy(indices, vector, specialTag = null) {
    let finalResults = [];
    let usedIds = new Set();
    const SPECIAL_TAGS = [
        "Halloween",
        "Christmas",
        "Birthday",
        "Anniversary",
        "Travel",
        "Period",
        "Sick",
    ];

    console.log(
        `[Anima RAG] 🚀 开始精准分步检索 (库数量: ${indices.length})...`,
    );

    // ---------------------------------------------------------
    // Step 1: Global Top 2 (无视 Tags，纯靠向量分)
    // ---------------------------------------------------------
    // 每个库只要前 4 个，合并后取全网前 2 个
    const step1Results = await queryMultiIndices(indices, vector, 4);
    for (const res of step1Results) {
        if (finalResults.length >= 2) break;
        if (!usedIds.has(res.item.id)) {
            finalResults.push(res);
            usedIds.add(res.item.id);
        }
    }

    // 🕵️ 获取 Vibe
    let vibeLabelA = null;
    if (finalResults.length > 0) {
        const topTags = finalResults[0].item.metadata.tags || [];
        vibeLabelA = topTags.find(
            (t) =>
                t !== "Important" &&
                !SPECIAL_TAGS.some(
                    (st) => st.toLowerCase() === t.toLowerCase(),
                ),
        );
    }
    console.log(`[Step 1] 选出 Top 2. Vibe 锁定: ${vibeLabelA || "无"}`);

    // ---------------------------------------------------------
    // Step 2: Important (Top 1) - 数据库级精准过滤
    // ---------------------------------------------------------
    // 直接让 Vectra 只在 Important 里找，哪怕分再低也能找到！
    const importantCandidates = await queryMultiIndices(indices, vector, 3, {
        tags: { $in: ["Important"] },
    });

    const importantMatch = importantCandidates.find(
        (r) => !usedIds.has(r.item.id),
    );
    if (importantMatch) {
        finalResults.push(importantMatch);
        usedIds.add(importantMatch.item.id);
        console.log(
            `[Step 2] 精准捕获 Important: ID=${importantMatch.item.id}`,
        );
    }

    // ---------------------------------------------------------
    // Step 3: Diversity (Top 2) - 数据库级精准排除
    // ---------------------------------------------------------
    // 告诉 Vectra: 不要 Important，不要 Vibe A。剩下的给我按分排！
    const excludeTags = ["Important"];
    if (vibeLabelA) excludeTags.push(vibeLabelA);

    const diversityCandidates = await queryMultiIndices(indices, vector, 6, {
        tags: { $nin: excludeTags },
    });

    let addedRichness = 0;
    for (const r of diversityCandidates) {
        if (addedRichness >= 2) break;
        if (!usedIds.has(r.item.id)) {
            finalResults.push(r);
            usedIds.add(r.item.id);
            addedRichness++;
        }
    }
    console.log(`[Step 3] 捕获多样性切片: ${addedRichness} 个`);

    // ---------------------------------------------------------
    // Step 4: Special (Top 1) - 数据库级精准过滤
    // ---------------------------------------------------------
    if (specialTag) {
        // 哪怕这个 Special 切片的相关性只有 0.001，Vectra 也会把它翻出来
        const specialCandidates = await queryMultiIndices(indices, vector, 3, {
            tags: { $in: [specialTag] },
        });
        const specialMatch = specialCandidates.find(
            (r) => !usedIds.has(r.item.id),
        );
        if (specialMatch) {
            finalResults.push(specialMatch);
            usedIds.add(specialMatch.item.id);
            console.log(
                `[Step 4] 精准捕获 Special (${specialTag}): ID=${specialMatch.item.id}`,
            );
        }
    }

    // ---------------------------------------------------------
    // Final Sort (按时间 > 切片序号)
    // ---------------------------------------------------------
    finalResults.sort((a, b) => {
        const itemA = a.item.metadata;
        const itemB = b.item.metadata;
        const timeA = new Date(itemA.timestamp || 0).getTime();
        const timeB = new Date(itemB.timestamp || 0).getTime();
        if (timeA !== timeB) return timeA - timeB;

        const getSlice = (str) => parseInt((str || "0_0").split("_")[1] || 0);
        return getSlice(itemA.index) - getSlice(itemB.index);
    });

    return finalResults;
}

async function init(router) {
    if (!fs.existsSync(VECTOR_ROOT)) {
        fs.mkdirSync(VECTOR_ROOT, { recursive: true });
    }
    console.log("[Anima RAG] 向量存储根目录就绪:", VECTOR_ROOT);

    // API: 存入 (新增：写入前自动清理旧版本)
    router.post("/insert", async (req, res) => {
        const {
            collectionId,
            text,
            tags,
            timestamp,
            apiConfig,
            index,
            batch_id,
        } = req.body;

        try {
            await runInQueue(collectionId, async () => {
                const vector = await getEmbedding(text, apiConfig);
                const targetIndex = await getIndex(collectionId);

                // =========================================================
                // 🧹 步骤 0: 写入前自检，清理旧的同名 Index (防重复核心)
                // =========================================================
                if (index !== undefined && index !== null) {
                    const allItems = await targetIndex.listItems();

                    // 1. 找出旧的同名切片 (例如 index 为 "1_1" 的所有旧记录)
                    const duplicates = allItems.filter(
                        (item) =>
                            item.metadata &&
                            String(item.metadata.index) === String(index),
                    );

                    if (duplicates.length > 0) {
                        console.log(
                            `[Anima RAG] 🔄 更新检测: 发现 Index ${index} 的旧版本 ${duplicates.length} 个，正在覆盖...`,
                        );

                        // 2. 构建删除计划 (保存路径)
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
                        timestamp,
                        index,
                        batch_id: parseInt(batch_id),
                    },
                });

                console.log(
                    `[Anima RAG] ✅ 写入成功 | Batch: ${batch_id} | Index: ${index}`,
                );
                res.json({ success: true, vectorId: newItem.id });
            });
        } catch (err) {
            console.error("[Anima RAG Insert Error]", err);
            if (!res.headersSent) res.status(500).send(err.message);
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

    // API: 查询
    router.post("/query", async (req, res) => {
        try {
            let {
                searchText,
                specialTag,
                apiConfig,
                collectionId,
                collectionIds,
            } = req.body;

            // 兼容数组和单值
            let targetIds = [];
            if (Array.isArray(collectionIds)) targetIds = collectionIds;
            else if (collectionId) targetIds = [collectionId];

            // 过滤空值
            targetIds = targetIds.filter((id) => id);

            if (targetIds.length === 0) return res.json([]);

            const vector = await getEmbedding(searchText, apiConfig);

            // 并行加载库
            const indices = (
                await Promise.all(
                    targetIds.map((id) => getIndex(id).catch((e) => null)),
                )
            ).filter((i) => i !== null);

            if (indices.length === 0) return res.json([]);

            // 执行策略
            const results = await perform2121Strategy(
                indices,
                vector,
                specialTag,
            );

            const responseData = results.map((r) => ({
                text: r.item.metadata.text,
                tags: r.item.metadata.tags,
                score: r.score,
                timestamp: r.item.metadata.timestamp,
                index: r.item.metadata.index,
                batch_id: r.item.metadata.batch_id,
            }));
            res.json(responseData);
        } catch (err) {
            console.error(err);
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
