# Who gets a mortgage? Interactive XAI app

Streamlit app for the executive XAI training. Real HMDA mortgage data (Maryland 2023), one gradient-boosting model,
one step per page. New steps are added as the course develops.

## Run it on your laptop (also the backup on the day)
```bash
cd xai_app
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

## Put it online (Streamlit Community Cloud, free)
1. Create a new **GitHub repository** (public is fine: the data is public) and upload the contents of this folder.
2. Go to **share.streamlit.io**, sign in with GitHub, click **Create app**, pick the repository, main file `app.py`.
3. Under **Advanced settings** choose **Python 3.13** (the version it was tested with) and paste the unlock codes
   into **Secrets** (see below).
4. Deploy. You get a permanent link; turn it into a QR code for the slides.

**On the day:** open the link yourself ~30 minutes before the session (free apps sleep when unused and need ~1 minute to wake).
**Do not change the repository during class**: every change redeploys the app.

## Unlock codes
- Locked steps ask for a code you announce in class. The codes are **not in this repository** (it is public):
  they live in Streamlit secrets, locally in `.streamlit/secrets.toml` (git-ignored) and online in
  Streamlit Cloud → app settings → **Secrets**, in this format:
  ```toml
  presenter_key = "choose-a-password"

  [unlock_codes]
  importance = "..."   # step 3
  pdp = "..."          # step 4
  shap = "..."         # step 5
  stable = "..."       # step 6
  sources = "..."      # Where did the answer come from?
  trace = "..."        # Agent detective
  ```
- A locked step without a code in secrets stays locked. To lock a new step: call `core.unlocked("key")` in it,
  add the key to `LOCKED_PAGES` in `core.py`, and add its code to secrets.
- **Presenter view:** add `?presenter=<presenter_key>` to the URL to open every step (for rehearsing or projecting).

## Structure
| Path | What |
|---|---|
| `app.py` | Menu. To add a step: add one `st.Page(...)` line |
| `core.py` | Shared data, model, the 10 applicants, styling, unlock codes |
| `steps/` | One file per step |
| `data/` | Train / test data (HMDA Maryland 2023, cleaned) |
| `artifacts/` | Saved model, the 10 fixed applicants, precomputed importances |
| `make_shap_plots.py` | Renders the SHAP waterfall / beeswarm / bar images in `artifacts/shap_plots/` (run after `precompute.py`) |
| `precompute.py` | Recomputes `artifacts/importance.json`, `artifacts/ice.npz`, `artifacts/shap.npz` and `artifacts/stability.npz` (needs `pip install shap`; the app itself does not) (run after changing the model) |

The model loads from `artifacts/credit_model.pkl`. If the installed scikit-learn version differs, the app retrains
the identical model once at start-up (~30 s), so it never breaks on a version mismatch.
