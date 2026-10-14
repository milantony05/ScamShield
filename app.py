from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from core import warning_cues, explain


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ScamShield",
    page_icon="🛡️",
    layout="centered",
)

# Static presentation styles. User messages are not inserted into HTML.
st.markdown(
    """
    <style>
        .block-container {
            max-width: 960px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(120deg, #0F172A, #164E63);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
        }

        .hero .eyebrow {
            color: #5EEAD4;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.15em;
            margin-bottom: 0.75rem;
        }

        .hero h1 {
            color: white;
            font-size: 2.7rem;
            margin: 0;
            padding: 0;
        }

        .hero p {
            color: #CBD5E1;
            font-size: 1rem;
            margin: 0.8rem 0 0;
            max-width: 650px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #111C30;
            border: 1px solid #263449;
            border-radius: 16px;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 12px;
            font-size: 1rem;
            line-height: 1.6;
        }

        div[data-testid="stButton"] button[kind="primary"] {
            border-radius: 10px;
            min-height: 48px;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: #162238;
            border: 1px solid #2A3B52;
            border-radius: 12px;
            padding: 1rem;
        }

        div[data-testid="stMetricValue"] {
            color: #5EEAD4;
            font-size: 1.7rem;
        }

        div[data-testid="stAlert"]:has(
            div[data-testid="stAlertContentInfo"]
        ) {
            background-color: #10382B;
            color: #D1FAE5;
            border: 1px solid #34D399;
            border-radius: 12px;
        }

        div[data-testid="stAlertContentInfo"] {
            color: #D1FAE5;
        }

        .footer {
            text-align: center;
            color: #94A3B8;
            font-size: 0.8rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">PAUSE · CHECK · VERIFY</div>
        <h1>🛡️ ScamShield</h1>
        <p>
            A second look before your next click.
            Check a message for phishing patterns and understand
            what raised the warning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models/scamshield.joblib")


if not (ROOT / "models/scamshield.joblib").exists():
    st.error("The model is missing. Run: python train.py")
    st.stop()

bundle = load_model()

examples = json.loads(
    (ROOT / "data/demo_examples.json").read_text(encoding="utf-8")
)

with st.container(border=True):
    st.subheader("Check a message")
    st.caption(
        "Paste an English SMS or try one of the authored examples below."
    )

    choice = st.selectbox(
        "Choose an example",
        ["Write my own"] + [item["name"] for item in examples],
    )

    initial = next(
        (
            item["text"]
            for item in examples
            if item["name"] == choice
        ),
        "",
    )

    text = st.text_area(
        "Message",
        value=initial,
        placeholder="Paste the message you want to check...",
        height=180,
        max_chars=5000,
        key=choice,
    )

    st.caption(
        f"{len(text):,} characters · Processed locally · Links are not opened"
    )

    analyse = st.button(
        "Analyse message →",
        type="primary",
        width="stretch",
    )

if analyse:
    if not text.strip():
        st.warning("Paste a message or select an example first.")
    else:
        score = float(
            bundle["model"].predict_proba([text])[0, 1]
        )
        threshold = float(bundle["threshold"])
        flagged = score >= threshold

        st.write("")

        with st.container(border=True):
            st.subheader("Your analysis")

            if flagged:
                st.error(
                    "⚠️ Phishing suspected\n\n"
                    "This message resembles phishing patterns. "
                    "Verify independently before acting."
                )
            else:
                st.info(
                    "🔎 No phishing pattern detected\n\n"
                    "The model did not detect a strong phishing pattern. "
                    "The message’s authenticity remains unverified."
                )

            left, right = st.columns(2)

            with left:
                st.metric("Model score", f"{score:.3f}")

            with right:
                st.metric("Warning threshold", f"{threshold:.3f}")

            st.caption(
                "A warning appears when the score reaches the threshold. "
                "The score is not a verified probability of fraud."
            )

            cues = warning_cues(text)

            st.markdown("**Warning cues in this message**")

            if cues:
                for cue in cues:
                    st.write(f"• {cue}")
            else:
                st.write("No predefined warning cues matched.")

            st.caption(
                "These are separate rule-based cues. "
                "A matched cue alone does not prove fraud."
            )

            with st.expander("Why did the model give this score?"):
                evidence = explain(bundle["model"], text)

                if evidence:
                    evidence_df = pd.DataFrame(evidence)

                    evidence_df["feature"] = (
                        evidence_df["feature"]
                        .str.replace("structure__", "", regex=False)
                        .str.replace("word__", "", regex=False)
                        .str.replace("char__", "", regex=False)
                    )

                    evidence_df = evidence_df.rename(
                        columns={
                            "feature": "Feature",
                            "contribution": "Contribution",
                        }
                    )

                    st.dataframe(
                        evidence_df,
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.write(
                        "No positive feature contributions to display."
                    )

                st.caption(
                    "Positive contributions increase the model’s "
                    "phishing score. They show learned associations, "
                    "not proof of fraud."
                )

            st.divider()
            st.markdown("**Before you act**")
            st.write(
                "Verify through the service’s official app or a "
                "contact channel you already trust. Do not disclose "
                "OTPs, PINs or passwords to a sender."
            )

with st.expander("About ScamShield"):
    st.write(
        "ScamShield is an academic SMS phishing screening prototype "
        "trained on a public English-message benchmark."
    )
    st.write(
        "It does not authenticate senders, inspect linked websites, "
        "or detect every type of financial scam."
    )
    st.write(
        "India-focused examples are manually authored demonstrations, "
        "not an independent evaluation dataset."
    )
    st.caption(
        "Submitted messages are processed locally and are not "
        "deliberately saved by this app."
    )

st.markdown(
    """
    <div class="footer">
        ScamShield · Explainable message screening · Academic prototype
    </div>
    """,
    unsafe_allow_html=True,
)