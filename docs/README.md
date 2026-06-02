# PRISM documentation (static site)

Open **`index.html`** in a browser, or serve locally:

```bash
cd docs
python -m http.server 8080
# http://localhost:8080
```

## GitHub

These files are **not** ignored by `.gitignore` (see `!docs/**` rules in the repo root). After editing, commit and push:

```bash
git add docs/
git commit -m "Update documentation site"
git push
```

Browse on GitHub: `docs/index.html` in the file tree, or enable **GitHub Pages** (Settings → Pages → Source: **Deploy from branch** → Folder: **`/docs`**) to get a live URL at `https://<user>.github.io/<repo>/`.
