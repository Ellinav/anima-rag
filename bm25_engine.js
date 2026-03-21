const fs = require("fs");
const path = require("path");
const MiniSearch = require("minisearch");
const { add_word, cut_for_search } = require("jieba-wasm");

const BM25_ROOT = path.join(__dirname, "data", "bm25_indexes");

class BM25Engine {
    constructor() {
        if (!fs.existsSync(BM25_ROOT)) {
            fs.mkdirSync(BM25_ROOT, { recursive: true });
        }
        // 内存缓存：dbId -> MiniSearch Instance
        this.activeIndexes = new Map();
    }

    _syncJieba(dictionary) {
        if (!dictionary || dictionary.length === 0) return;
        dictionary.forEach((rule) => {
            // 🟢 提取所有触发词 (兼容各种格式)
            let safeTriggers = [];
            if (Array.isArray(rule.triggers)) safeTriggers = rule.triggers;
            else if (typeof rule.trigger === "string")
                safeTriggers = rule.trigger.split(/[,，|]/);
            else if (rule.triggers) safeTriggers = [rule.triggers];

            // 🟢 提取所有索引词 (兼容各种格式)
            let safeIndexWords = [];
            if (Array.isArray(rule.indexWords))
                safeIndexWords = rule.indexWords;
            else if (typeof rule.index === "string")
                safeIndexWords = rule.index.split(/[,，|]/);
            else if (rule.indexWords) safeIndexWords = [rule.indexWords];

            // 🌟 合并词池：让分词器认识所有的触发词和索引词
            const allWords = [...new Set([...safeTriggers, ...safeIndexWords])];

            allWords.forEach((word) => {
                const cleanWord = word?.trim();
                if (cleanWord) {
                    // ✅ 将 nodejieba.insertWord 替换为 add_word
                    add_word(cleanWord);
                }
            });
        });
    }

    // 🛠️ 辅助：中文分词器 + 强力停用词过滤 (注入给 MiniSearch)
    _tokenize(text) {
        if (!text) return [];

        const rawTokens = cut_for_search(text, true);

        const stopWords = new Set([
            // === 1. 代词与指代 (极高频，毫无区分度) ===
            "我",
            "你",
            "他",
            "她",
            "它",
            "我们",
            "你们",
            "他们",
            "他们",
            "它们",
            "自己",
            "这",
            "那",
            "什么",
            "怎么",
            "哪个",
            "这个",
            "那个",
            "哪些",
            "这些",
            "那些",

            // === 2. 助词与语气词 (句法骨架与口癖废话) ===
            "的",
            "了",
            "着",
            "过",
            "得",
            "地",
            "啊",
            "呀",
            "吧",
            "呢",
            "哦",
            "吗",
            "哈",
            "嗯",
            "哎",
            "咦",
            "诶",
            "咯",

            // === 3. 连词、介词与方位词 (逻辑与空间连接) ===
            "和",
            "并",
            "以",
            "而",
            "及",
            "与",
            "或",
            "在",
            "把",
            "被",
            "让",
            "对",
            "到",
            "上",
            "下",
            "中",
            "里",
            "将",
            "因",
            "因为",
            "所以",
            "如果",
            "虽然",
            "可是",
            "但是",
            "然而",
            "就是",
            "还是",
            "但",
            "却",
            "因此",
            "于是",
            "不过",
            "然后",
            "接着",

            // === 4. 程度副词、频次与时间 (削弱强度的修饰词) ===
            "就",
            "都",
            "才",
            "又",
            "也",
            "会",
            "能",
            "不",
            "已",
            "没有",
            "甚至",
            "居然",
            "仅",
            "只",
            "还有",
            "时候",
            "突然",
            "十分",
            "非常",
            "有些",
            "有点",
            "一点",
            "一下",

            // === 5. 基础状态与高频泛用动词 (缺乏实体指向性) ===
            "是",
            "有",
            "去",
            "来",
            "用",
            "知道",
            "觉得",
            "感觉",
            "开始",
            "准备",
            "继续",

            // === 6. 量词与文本切割残留 ===
            "个",
            "一段",
            "一些",
            "一种",
            "一个",
            "一丝",
            "极其",

            // === 8. 🎭 RP 专属：环境神态与主观臆测 (防氛围污染) ===
            "微微",
            "轻轻",
            "缓缓",
            "似乎",
            "好像",
            "也许",
            "可能",
            "其实",
            "仿佛",
            "不禁",
            "忍不住",
        ]);

        return rawTokens.filter((token) => {
            const t = token.trim();
            if (!t) return false;
            if (stopWords.has(t)) return false;

            // 🟢 核心修复：只要不包含"中文"或"英文字母"，就统统干掉（专杀纯标点、纯数字）
            if (!/[\u4e00-\u9fa5a-zA-Z]/.test(t)) return false;

            return true;
        });
    }

    // 🛠️ 辅助：RRF (Reciprocal Rank Fusion) 倒数排名融合
    _applyRRF(multiDbResults, k = 60) {
        const rrfMap = new Map();

        multiDbResults.forEach((dbResults) => {
            dbResults.forEach((item, index) => {
                const rank = index + 1;
                const score = 1 / (k + rank);

                if (rrfMap.has(item.id)) {
                    const existing = rrfMap.get(item.id);
                    existing.rrfScore += score;
                } else {
                    rrfMap.set(item.id, {
                        ...item,
                        rrfScore: score,
                    });
                }
            });
        });

        // 拍平并按 RRF 分数降序
        return Array.from(rrfMap.values()).sort(
            (a, b) => b.rrfScore - a.rrfScore,
        );
    }

    // 🛠️ 辅助：解析 ID 用于最终排序
    _parseId(indexStr) {
        const parts = (indexStr || "0_0").split("_");
        return {
            batch: parseInt(parts[0] || 0),
            slice: parseInt(parts[1] || 0),
        };
    }
    _getSafeDbId(dbId) {
        if (!dbId) return "unknown_db";
        return dbId.replace(/[^a-zA-Z0-9@\-\._\u4e00-\u9fa5]/g, "_");
    }

    async buildIndex(dbId, chunks, dictionary = [], type = "chat") {
        // ✨ 新增：建库前，先让分词器把词典里的词学会
        this._syncJieba(dictionary);

        // 1. 初始化 MiniSearch
        const miniSearch = new MiniSearch({
            fields: ["searchableText"],
            storeFields: [
                "id",
                "text",
                "tags",
                "timestamp",
                "index",
                "batch_id",
                "chunk_index",
                "doc_name",
                "_source_db",
            ],
            tokenize: this._tokenize,
            // BM25 参数调优 (可根据后续效果微调)
            bm25: { k1: 1.2, b: 0.75 },
        });

        const documents = chunks.map((chunk) => {
            let searchableParts = [chunk.text];
            let finalTags = []; // 🟢 BM25专属标签池，完全由词典决定！

            dictionary.forEach((rule) => {
                // 🟢 兼容前端的 { trigger: "A,B", index: "C" } 格式
                let safeTriggers = [];
                if (Array.isArray(rule.triggers)) safeTriggers = rule.triggers;
                else if (typeof rule.trigger === "string")
                    safeTriggers = rule.trigger.split(/[,，|]/);

                let safeIndexWords = [];
                if (Array.isArray(rule.indexWords))
                    safeIndexWords = rule.indexWords;
                else if (typeof rule.index === "string")
                    safeIndexWords = rule.index.split(/[,，|]/);

                // 清洗首尾空格并过滤空值
                safeTriggers = safeTriggers
                    .map((t) => t?.trim())
                    .filter(Boolean);
                safeIndexWords = safeIndexWords
                    .map((w) => w?.trim())
                    .filter(Boolean);

                // 兜底逻辑：如果用户没填触发词，默认用索引词做触发词
                const combinedTriggers = [
                    ...new Set([...safeTriggers, ...safeIndexWords]),
                ];

                const hasTrigger = combinedTriggers.some((trigger) =>
                    chunk.text.includes(trigger),
                );

                if (hasTrigger) {
                    searchableParts.push(...safeIndexWords);
                    finalTags.push(...safeIndexWords);
                }
            });

            return {
                ...chunk,
                // 🛑 核心修改：彻底抛弃原本的 chunk.tags，只用词典命中生成的 finalTags
                tags: finalTags.length > 0 ? [...new Set(finalTags)] : [],
                _source_db: dbId,
                searchableText: searchableParts.join(" "),
            };
        });

        miniSearch.addAll(documents);

        const safeDbId = this._getSafeDbId(dbId); // 🟢 统一清洗
        // 存入内存与本地盘 (都用 safeDbId)
        this.activeIndexes.set(safeDbId, miniSearch);
        const indexPath = path.join(BM25_ROOT, `${safeDbId}.json`);
        fs.writeFileSync(indexPath, JSON.stringify(miniSearch.toJSON()));

        console.log(
            `[Anima BM25] 📚 构建索引成功: ${safeDbId} | 条目数: ${documents.length}`,
        );
        return true;
    }

    /**
     * 🚀 极速批处理：一次性将多个切片写入 BM25 库，仅执行 1 次 IO
     */
    async buildIndexBatch(dbId, slices, bm25Config) {
        const safeDbId = this._getSafeDbId(dbId);
        const indexPath = path.join(BM25_ROOT, `${safeDbId}.json`);

        this._syncJieba(bm25Config?.dictionary);

        let miniSearch;
        if (this.activeIndexes.has(safeDbId)) {
            miniSearch = this.activeIndexes.get(safeDbId);
        } else if (fs.existsSync(indexPath)) {
            const data = fs.readFileSync(indexPath, "utf8");
            miniSearch = MiniSearch.loadJSON(data, {
                fields: ["text"],
                storeFields: [
                    "id",
                    "text",
                    "tags",
                    "timestamp",
                    "index",
                    "batch_id",
                ],
                tokenize: this._tokenize,
            });
            this.activeIndexes.set(safeDbId, miniSearch);
        } else {
            miniSearch = new MiniSearch({
                fields: ["searchableText"], // ⚠️ 修复：必须是 searchableText
                storeFields: [
                    "id",
                    "text",
                    "tags",
                    "timestamp",
                    "index",
                    "batch_id",
                    "chunk_index",
                    "doc_name",
                    "_source_db",
                ],
                tokenize: this._tokenize,
                bm25: { k1: 1.2, b: 0.75 },
            });
            this.activeIndexes.set(safeDbId, miniSearch);
        }

        let addedCount = 0;
        const dictionary = bm25Config?.dictionary || [];

        // 3. 内存中批量分词与索引构建
        for (const slice of slices) {
            // ⚠️ 修复：把单条插入的词典判断逻辑补回来！
            let searchableParts = [slice.text];
            let finalTags = [];

            dictionary.forEach((rule) => {
                let safeTriggers = Array.isArray(rule.triggers)
                    ? rule.triggers
                    : (rule.trigger || "").split(/[,，|]/);
                let safeIndexWords = Array.isArray(rule.indexWords)
                    ? rule.indexWords
                    : (rule.index || "").split(/[,，|]/);

                safeTriggers = safeTriggers
                    .map((t) => t?.trim())
                    .filter(Boolean);
                safeIndexWords = safeIndexWords
                    .map((w) => w?.trim())
                    .filter(Boolean);
                const combinedTriggers = [
                    ...new Set([...safeTriggers, ...safeIndexWords]),
                ];

                const hasTrigger = combinedTriggers.some((trigger) =>
                    slice.text.includes(trigger),
                );
                if (hasTrigger) {
                    searchableParts.push(...safeIndexWords);
                    finalTags.push(...safeIndexWords);
                }
            });

            const doc = {
                id: String(slice.index),
                index: slice.index,
                text: slice.text,
                tags: finalTags.length > 0 ? [...new Set(finalTags)] : [],
                timestamp: slice.timestamp || Date.now(),
                batch_id: slice.batch_id || 0,
                searchableText: searchableParts.join(" "), // ⚠️ 修复：拼装隐藏搜索域
                _source_db: safeDbId,
            };

            if (miniSearch.has(doc.id)) miniSearch.discard(doc.id);
            miniSearch.add(doc);
            addedCount++;
        }

        // 4. 仅在此刻执行唯一一次硬盘写入
        if (addedCount > 0) {
            fs.writeFileSync(indexPath, JSON.stringify(miniSearch.toJSON()));
            console.log(
                `[Anima BM25] ⚡ 批量增量写入成功: ${safeDbId} | 共 ${addedCount} 条切片`,
            );
        }

        return addedCount;
    }

    async upsertDocument(dbId, chunk, dictionary = [], type = "chat") {
        this._syncJieba(dictionary);

        const safeDbId = this._getSafeDbId(dbId); // 🟢 统一清洗
        let miniSearch = this.activeIndexes.get(safeDbId);
        const indexPath = path.join(BM25_ROOT, `${safeDbId}.json`);

        // 1. 如果内存没有，尝试从本地加载或新建
        if (!miniSearch) {
            if (fs.existsSync(indexPath)) {
                const indexData = fs.readFileSync(indexPath, "utf-8");
                miniSearch = MiniSearch.loadJSON(indexData, {
                    fields: ["searchableText"],
                    storeFields: [
                        "id",
                        "text",
                        "tags",
                        "timestamp",
                        "index",
                        "batch_id",
                        "chunk_index",
                        "doc_name",
                        "_source_db",
                    ],
                    tokenize: this._tokenize,
                });
            } else {
                miniSearch = new MiniSearch({
                    fields: ["searchableText"],
                    storeFields: [
                        "id",
                        "text",
                        "tags",
                        "timestamp",
                        "index",
                        "batch_id",
                        "chunk_index",
                        "doc_name",
                        "_source_db",
                    ],
                    tokenize: this._tokenize,
                    bm25: { k1: 1.2, b: 0.75 },
                });
            }
            this.activeIndexes.set(safeDbId, miniSearch);
        }

        // 2. 处理单条数据
        let searchableParts = [chunk.text];
        let finalTags = [];

        dictionary.forEach((rule) => {
            // 🟢 兼容前端的 { trigger: "A,B", index: "C" } 格式
            let safeTriggers = [];
            if (Array.isArray(rule.triggers)) safeTriggers = rule.triggers;
            else if (typeof rule.trigger === "string")
                safeTriggers = rule.trigger.split(/[,，|]/);

            let safeIndexWords = [];
            if (Array.isArray(rule.indexWords))
                safeIndexWords = rule.indexWords;
            else if (typeof rule.index === "string")
                safeIndexWords = rule.index.split(/[,，|]/);

            safeTriggers = safeTriggers.map((t) => t?.trim()).filter(Boolean);
            safeIndexWords = safeIndexWords
                .map((w) => w?.trim())
                .filter(Boolean);

            const combinedTriggers = [
                ...new Set([...safeTriggers, ...safeIndexWords]),
            ];

            const hasTrigger = combinedTriggers.some((trigger) =>
                chunk.text.includes(trigger),
            );

            if (hasTrigger) {
                searchableParts.push(...safeIndexWords);
                finalTags.push(...safeIndexWords);
            }
        });

        const document = {
            ...chunk,
            // 🛑 核心修改：覆盖原 tags
            tags: finalTags.length > 0 ? [...new Set(finalTags)] : [],
            _source_db: safeDbId,
            searchableText: searchableParts.join(" "),
        };

        // 3. 覆盖或新增 (如果 id 已存在会自动替换，防止重复)
        if (miniSearch.has(document.id)) {
            miniSearch.replace(document);
        } else {
            miniSearch.add(document);
        }

        // 4. 保存到本地磁盘
        fs.writeFileSync(indexPath, JSON.stringify(miniSearch.toJSON()));
        console.log(
            `[Anima BM25] 📝 增量写入成功: ${dbId} | Index: ${chunk.index}`,
        );
        return true;
    }

    /**
     * 🧠 核心 2：多库触发与 RRF 检索 Pipeline
     * @param {string} queryText - 用户的输入或近期楼层文本
     * @param {Array} dbConfigs - [{ dbId: 'chat_A', dictionary: [...] }, ...]
     * @param {number} topK - 截取数量
     * @param {string} type - 'chat' | 'kb'
     */
    async searchPipeline(queryText, dbConfigs, topK = 3, type = "chat") {
        if (!queryText || dbConfigs.length === 0) return [];

        // ✨ 新增：检索前，把本次检索涉及到的所有词典规则，都教给分词器
        dbConfigs.forEach((config) => this._syncJieba(config.dictionary));

        const multiDbResults = [];

        // --- 第一步：扫描触发词 & 执行检索 ---
        for (const config of dbConfigs) {
            const { dbId, dictionary } = config;
            const safeDbId = this._getSafeDbId(dbId); // 🟢 统一清洗

            // 加载库 (尝试从内存，没有则从硬盘)
            let miniSearch = this.activeIndexes.get(safeDbId);
            if (!miniSearch) {
                const indexPath = path.join(BM25_ROOT, `${safeDbId}.json`);
                if (fs.existsSync(indexPath)) {
                    const indexData = fs.readFileSync(indexPath, "utf-8");
                    miniSearch = MiniSearch.loadJSON(indexData, {
                        fields: ["searchableText"],
                        storeFields: [
                            "id",
                            "text",
                            "tags",
                            "timestamp",
                            "index",
                            "batch_id",
                            "chunk_index",
                            "doc_name",
                            "_source_db",
                        ],
                        tokenize: this._tokenize,
                    });
                    this.activeIndexes.set(safeDbId, miniSearch);
                } else {
                    console.warn(`[Anima BM25] ⚠️ 库不存在，跳过: ${dbId}`);
                    continue;
                }
            }

            // 2. 检查触发词 (拦截器逻辑)
            let matchedIndexWords = [];
            let isTriggered = false;

            if (dictionary && dictionary.length > 0) {
                dictionary.forEach((rule) => {
                    // 🟢 完美兼容：同时支持前端传来的 trigger/triggers, index/indexWords
                    let safeTriggers = [];
                    if (Array.isArray(rule.triggers))
                        safeTriggers = rule.triggers;
                    else if (typeof rule.trigger === "string")
                        safeTriggers = rule.trigger.split(/[,，|]/);
                    else if (rule.triggers) safeTriggers = [rule.triggers];

                    let safeIndexWords = [];
                    if (Array.isArray(rule.indexWords))
                        safeIndexWords = rule.indexWords;
                    else if (typeof rule.index === "string")
                        safeIndexWords = rule.index.split(/[,，|]/);
                    else if (rule.indexWords)
                        safeIndexWords = [rule.indexWords];

                    // 将触发词和索引词合并成一个巨大的触发检测池
                    let combinedForTrigger = [
                        ...safeTriggers,
                        ...safeIndexWords,
                    ];

                    const hit = combinedForTrigger.some((trigger) => {
                        if (!trigger || String(trigger).trim() === "")
                            return false;
                        const cleanTrigger = String(trigger)
                            .trim()
                            .toLowerCase();
                        return queryText.toLowerCase().includes(cleanTrigger);
                    });

                    if (hit) {
                        matchedIndexWords.push(
                            ...safeIndexWords
                                .map((w) => String(w).trim())
                                .filter((w) => w !== ""),
                        );
                        isTriggered = true;
                    }
                });

                // 🛑 核心规则：如果没命中任何触发词，则完全跳过对应库的 BM25 检索
                if (!isTriggered) {
                    console.log(
                        `[Anima BM25] 🛑 过滤跳过: ${dbId} (未命中任何触发词)`,
                    );
                    continue;
                }
            } else {
                // 如果该库没有配置词典，默认允许检索
                isTriggered = true;
            }

            // 3. 构建联合查询词：原句关键信息 + 提取到的索引词
            // 原句也送进去，依靠 nodejieba 分词去匹配 searchableText
            const finalQueryStr = `${queryText} ${matchedIndexWords.join(" ")}`;

            // 4. 执行单库检索 (关闭 fuzzy 和 prefix，防止无意义发散匹配)
            let results = miniSearch.search(finalQueryStr, {
                prefix: false,
                combineWith: "OR",
            });

            // 🌟 核心修复：精准数组比对与平滑加权
            if (results.length > 0 && matchedIndexWords.length > 0) {
                results = results.filter((res) => {
                    // 1. 获取底层切片真实的 tags 数组 (我们在入库时打的钢印)
                    const sliceTags = res.tags || [];
                    let exactTagHitCount = 0;

                    // 2. 严格核对：tags 里是否真的包含索引词
                    matchedIndexWords.forEach((indexWord) => {
                        const lowerIndex = indexWord.toLowerCase();
                        if (
                            sliceTags.some(
                                (tag) => tag.toLowerCase() === lowerIndex,
                            )
                        ) {
                            exactTagHitCount++;
                        }
                    });

                    // 3. 【理智的加权算法】平滑的线性加权
                    // 比如命中 1 个索引词：分数 * 1.5 倍
                    // 命中 2 个索引词：分数 * 2.0 倍
                    if (exactTagHitCount > 0) {
                        res.score = res.score * (1 + 0.5 * exactTagHitCount);
                    }

                    // 不强制 return false，让没命中索引词的切片也保留正常的基础分参与竞选
                    return true;
                });

                // 提权完毕后，重新排序
                results.sort((a, b) => b.score - a.score);

                // 局部截断：每个库只提供自己的 Top K
                results = results.slice(0, topK);
            }

            if (results.length > 0) {
                multiDbResults.push(results);
            }
        }

        if (multiDbResults.length === 0) return [];

        // --- 第二步：全局分数融合与截断 ---
        // 1. 拍平所有库提交的局部 Top K
        let fusedResults = multiDbResults.flat();

        // 2. 全局绝对降序
        fusedResults.sort((a, b) => b.score - a.score);

        // 3. 严格截取全局 Top K
        fusedResults = fusedResults.slice(0, topK);

        // --- 第三步：最终输出的时间线重排 ---
        if (type === "chat") {
            fusedResults.sort((a, b) => {
                const timeA = new Date(a.timestamp || 0).getTime();
                const timeB = new Date(b.timestamp || 0).getTime();

                if (timeA > 0 && timeB > 0 && timeA !== timeB) {
                    return timeA - timeB;
                }

                const idA = this._parseId(a.index);
                const idB = this._parseId(b.index);

                if (idA.batch !== idB.batch) {
                    return idA.batch - idB.batch;
                }
                return idA.slice - idB.slice;
            });
        }

        const summaryDetail = fusedResults
            .map(
                (r) =>
                    `[📦${r._source_db}] #${r.index !== undefined ? r.index : r.chunk_index !== undefined ? r.chunk_index : "N/A"}`,
            )
            .join(" | ");
        console.log(
            `[Anima BM25] 🎯 全局检索完成 | 类型: ${type} | 返回 ${fusedResults.length} 条 -> ${summaryDetail || "无"}`,
        );
        return fusedResults;
    }

    /**
     * 🧠 核心 3：时间极值意图拦截器 (智能降级版)
     */
    async temporalIntentSearch(entities, dbConfigs, intentType) {
        if (!entities || entities.length === 0 || dbConfigs.length === 0)
            return [];

        dbConfigs.forEach((config) => this._syncJieba(config.dictionary));

        let allCandidateDocs = [];

        // 1. 扫描所有允许的库，把包含这些实体的切片全部捞出来
        for (const config of dbConfigs) {
            const safeDbId = this._getSafeDbId(config.dbId);
            let miniSearch = this.activeIndexes.get(safeDbId);

            if (!miniSearch) {
                const indexPath = path.join(BM25_ROOT, `${safeDbId}.json`);
                if (fs.existsSync(indexPath)) {
                    const indexData = fs.readFileSync(indexPath, "utf-8");
                    miniSearch = MiniSearch.loadJSON(indexData, {
                        fields: ["searchableText"],
                        storeFields: [
                            "id",
                            "text",
                            "tags",
                            "timestamp",
                            "index",
                            "batch_id",
                            "chunk_index",
                            "doc_name",
                            "_source_db",
                        ],
                        tokenize: this._tokenize,
                    });
                    this.activeIndexes.set(safeDbId, miniSearch);
                } else {
                    continue;
                }
            }

            // 用 OR 逻辑粗筛
            const results = miniSearch.search(entities.join(" "), {
                prefix: false,
                combineWith: "OR",
            });

            // 严格核对 Tags (防误杀：只收录真实验明正身的切片)
            results.forEach((res) => {
                const docTags = res.tags || [];
                const matchedEntities = entities.filter((e) =>
                    docTags.some((t) => t.toLowerCase() === e.toLowerCase()),
                );

                if (matchedEntities.length > 0) {
                    allCandidateDocs.push({
                        ...res,
                        _matched_entities: matchedEntities,
                        _source_db: safeDbId,
                    });
                }
            });
        }

        if (allCandidateDocs.length === 0) return [];

        // 2. 执行绝对的物理时间线排序 (无视文本算分！)
        allCandidateDocs.sort((a, b) => {
            const timeA = new Date(a.timestamp || 0).getTime();
            const timeB = new Date(b.timestamp || 0).getTime();

            if (timeA > 0 && timeB > 0 && timeA !== timeB) {
                return timeA - timeB; // 升序：最早的在前面
            }

            const idA = this._parseId(a.index);
            const idB = this._parseId(b.index);

            if (idA.batch !== idB.batch) return idA.batch - idB.batch;
            return idA.slice - idB.slice;
        });

        // 如果意图是“最后一次”，反转数组，让最新的排在最上面
        if (intentType === "last") {
            allCandidateDocs.reverse();
        }

        // 3. 🧠 智能降级提取策略 (Smart Fallback)
        // 策略 A：寻找“完美同框” (该切片包含了用户提到的所有实体)
        const perfectMatchDoc = allCandidateDocs.find(
            (doc) => doc._matched_entities.length === entities.length,
        );

        if (perfectMatchDoc) {
            console.log(
                `[Anima BM25] 🎯 意图拦截: 成功找到 [${entities.join(" + ")}] 的完美同框切片!`,
            );
            return [perfectMatchDoc];
        }

        // 策略 B：降级拆分提取 (没有同框，就各自找第一条/最后一条)
        console.log(
            `[Anima BM25] ⚠️ 意图拦截: 未找到完美同框，自动降级为独立提取...`,
        );
        let fallbackResults = [];
        let foundEntities = new Set();

        for (const doc of allCandidateDocs) {
            // 检查这个切片包含了哪些我们还没找到的实体
            const usefulFor = doc._matched_entities.filter(
                (e) => !foundEntities.has(e),
            );

            if (usefulFor.length > 0) {
                fallbackResults.push(doc);
                usefulFor.forEach((e) => foundEntities.add(e));
            }
            // 如果所有实体都凑齐了，下班！
            if (foundEntities.size === entities.length) break;
        }

        return fallbackResults;
    }

    /**
     * 🧠 核心 4：通过 ID 批量/单条删除 BM25 数据
     */
    async deleteDocuments(dbId, ids = []) {
        if (!ids || ids.length === 0) return;

        const safeDbId = this._getSafeDbId(dbId); // 🟢 统一清洗
        let miniSearch = this.activeIndexes.get(safeDbId);
        const indexPath = path.join(BM25_ROOT, `${safeDbId}.json`);

        if (!miniSearch) {
            if (!fs.existsSync(indexPath)) return; // 连文件都没有，说明没东西可删

            const indexData = fs.readFileSync(indexPath, "utf-8");
            miniSearch = MiniSearch.loadJSON(indexData, {
                fields: ["searchableText"],
                storeFields: [
                    "id",
                    "text",
                    "tags",
                    "timestamp",
                    "index",
                    "batch_id",
                    "chunk_index",
                    "doc_name",
                    "_source_db",
                ],
                tokenize: this._tokenize,
            });
            this.activeIndexes.set(safeDbId, miniSearch);
        }

        let changed = false;
        for (const id of ids) {
            // minisearch 提供了 discard 方法，可以安全地通过 ID 删除文档
            if (miniSearch.has(id)) {
                miniSearch.discard(id);
                changed = true;
            }
        }

        // 只有发生实际删除才写入硬盘，减少 IO
        if (changed) {
            fs.writeFileSync(indexPath, JSON.stringify(miniSearch.toJSON()));
            console.log(
                `[Anima BM25] 🗑️ 成功同步删除 ${dbId} 中的 ${ids.length} 条记录`,
            );
        }
    }

    /**
     * 🧠 核心 5：物理删除整个 BM25 库 (对应 /delete_collection)
     */
    async deleteIndex(dbId) {
        const safeDbId = this._getSafeDbId(dbId);
        if (this.activeIndexes.has(safeDbId)) {
            this.activeIndexes.delete(safeDbId);
        }
        const indexPath = path.join(BM25_ROOT, `${safeDbId}.json`);
        if (fs.existsSync(indexPath)) {
            fs.unlinkSync(indexPath);
            console.log(`[Anima BM25] 💥 彻底物理删除库: ${dbId}`);
        }
    }
}

module.exports = new BM25Engine();
