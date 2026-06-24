# nsfw-api

[![Rust](https://github.com/maa123/nsfw-api/actions/workflows/rust.yml/badge.svg)](https://github.com/maa123/nsfw-api/actions/workflows/rust.yml)

## 設定

ルートディレクトリに `config.toml` ファイルを作成してください。テンプレートとして `config.example.toml` を使用できます：

```bash
cp config.example.toml config.toml
```

`config.toml` を編集して、サーバーアドレスや `NC-time` ヘッダーの有無などの設定を行ってください。

`add_nc_time_header = false` を設定すると、レスポンスヘッダーへ `NC-time` を追加しません。
