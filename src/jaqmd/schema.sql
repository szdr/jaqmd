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

-- UPDATE: hash 変更時（active=1）→ FTS を差し替え
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
END;

-- DELETE: ハード削除時も FTS から削除
DROP TRIGGER IF EXISTS documents_delete;
CREATE TRIGGER documents_delete AFTER DELETE ON documents
WHEN OLD.active = 1 BEGIN
    DELETE FROM docs_fts_trigram WHERE docid = OLD.docid;
END;
