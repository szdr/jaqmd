-- コンテンツ実体（content-addressable、重複排除）
CREATE TABLE IF NOT EXISTS content (
    hash  TEXT PRIMARY KEY,
    body  TEXT NOT NULL
);

-- ドキュメント（ファイルパス → ハッシュのマッピング）
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    collection   TEXT NOT NULL,
    path         TEXT NOT NULL,
    hash         TEXT NOT NULL REFERENCES content(hash),
    docid        TEXT UNIQUE NOT NULL,
    title        TEXT,
    mtime        INTEGER,
    active       INTEGER DEFAULT 1,
    indexed_at   INTEGER DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection);
CREATE INDEX IF NOT EXISTS idx_documents_hash       ON documents(hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_path ON documents(collection, path);

-- コレクション定義
CREATE TABLE IF NOT EXISTS collections (
    id         INTEGER PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    path       TEXT NOT NULL,
    glob_mask  TEXT NOT NULL DEFAULT '**/*.md',
    created_at INTEGER DEFAULT (unixepoch())
);

-- パスコンテキスト
CREATE TABLE IF NOT EXISTS path_contexts (
    path    TEXT PRIMARY KEY,
    context TEXT NOT NULL
);

-- DBメタ情報（インデックス構築状態・モデル情報）
CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- FTS: trigram（trigram tokenizer、辞書不要）
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts_trigram USING fts5(
    docid    UNINDEXED,
    filepath UNINDEXED,
    title,
    body,
    tokenize = 'trigram'
);

-- FTS: morph（形態素解析、unicode61 tokenizer で分かち書き済みテキストを格納）
-- INSERT は Python 側（jaqmd morph）で実行。DELETE 系のみトリガーで同期。
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts_morph USING fts5(
    docid    UNINDEXED,
    filepath UNINDEXED,
    title,
    body,
    tokenize = 'unicode61'
);

-- INSERT 後: FTS に追加
DROP TRIGGER IF EXISTS documents_ai;
CREATE TRIGGER documents_ai AFTER INSERT ON documents
WHEN NEW.active = 1 BEGIN
    INSERT INTO docs_fts_trigram(docid, filepath, title, body)
    SELECT NEW.docid,
           NEW.collection || '/' || NEW.path,
           NEW.title,
           c.body
    FROM content c WHERE c.hash = NEW.hash;
END;

-- UPDATE: hash 変更時（active=1）→ trigram FTS を差し替え、morph FTS からは削除のみ
DROP TRIGGER IF EXISTS documents_au_hash;
CREATE TRIGGER documents_au_hash AFTER UPDATE ON documents
WHEN NEW.active = 1 AND NEW.hash != OLD.hash BEGIN
    DELETE FROM docs_fts_trigram WHERE docid = OLD.docid;
    INSERT INTO docs_fts_trigram(docid, filepath, title, body)
    SELECT NEW.docid,
           NEW.collection || '/' || NEW.path,
           NEW.title,
           c.body
    FROM content c WHERE c.hash = NEW.hash;
    DELETE FROM docs_fts_morph WHERE docid = OLD.docid;
END;

-- UPDATE: 再活性化（active 0→1、hash 変更なし）→ FTS に追加
DROP TRIGGER IF EXISTS documents_au_reactivate;
CREATE TRIGGER documents_au_reactivate AFTER UPDATE ON documents
WHEN OLD.active = 0 AND NEW.active = 1 AND NEW.hash = OLD.hash BEGIN
    INSERT INTO docs_fts_trigram(docid, filepath, title, body)
    SELECT NEW.docid,
           NEW.collection || '/' || NEW.path,
           NEW.title,
           c.body
    FROM content c WHERE c.hash = NEW.hash;
END;

-- UPDATE: 論理削除（active 1→0）→ FTS から削除
DROP TRIGGER IF EXISTS documents_soft_delete;
CREATE TRIGGER documents_soft_delete AFTER UPDATE ON documents
WHEN OLD.active = 1 AND NEW.active = 0 BEGIN
    DELETE FROM docs_fts_trigram WHERE docid = OLD.docid;
    DELETE FROM docs_fts_morph WHERE docid = OLD.docid;
END;

-- DELETE: ハード削除時も FTS から削除
DROP TRIGGER IF EXISTS documents_delete;
CREATE TRIGGER documents_delete AFTER DELETE ON documents
WHEN OLD.active = 1 BEGIN
    DELETE FROM docs_fts_trigram WHERE docid = OLD.docid;
    DELETE FROM docs_fts_morph WHERE docid = OLD.docid;
END;

-- チャンク本体（ベクトル検索用）
-- vectors_vec（vec0）は sqlite-vec 拡張ロード後に store.py で別途作成する。
CREATE TABLE IF NOT EXISTS chunk_vectors (
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES documents(id),
    docid       TEXT NOT NULL,
    chunk_seq   INTEGER NOT NULL,
    chunk_pos   INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    UNIQUE(docid, chunk_seq)
);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_docid ON chunk_vectors(docid);

-- Query Expansion キャッシュ（qe.py が szdr/jaqmd-qe-gemma-4-e2b-it の展開結果を保存）
CREATE TABLE IF NOT EXISTS qe_cache (
    query_hash TEXT PRIMARY KEY,
    query_raw  TEXT NOT NULL,
    lex_query  TEXT,
    vec_query  TEXT,
    hyde_text  TEXT,
    model_id   TEXT NOT NULL,
    created_at INTEGER DEFAULT (unixepoch()),
    ttl        INTEGER DEFAULT 86400
);
