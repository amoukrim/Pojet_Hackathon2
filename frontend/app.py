import streamlit as st
import plotly.graph_objects as go
import requests

st.title("🧠 Générateur de Texte + Filtrage Éthique")

API_URL = "http://localhost:8000"

if "stats" not in st.session_state:
    st.session_state.stats = {
        "total": 0,
        "passed": 0,
        "rejected": 0,
        "similarities": [],
        "causes": {}
    }

prompt = st.text_input("Entrez un prompt :", value="This movie was")

if st.button("Générer"):
    with st.spinner("Génération en cours..."):
        try:
            res = requests.post(f"{API_URL}/generate", json={"prompt": prompt})
            generated = res.json()["generated"]

            res = requests.post(f"{API_URL}/summarize", json={"prompt": generated})
            summary = res.json()["summary"]

            res = requests.post(f"{API_URL}/similarity", json={"prompt": prompt, "summary": summary})
            sim_score = res.json()["similarity"]
            is_relevant = sim_score > 0.3

            res = requests.post(f"{API_URL}/filter", json={"prompt": generated})
            filter_data = res.json()
            passes_filters = filter_data["passed"]
            reasons = filter_data["reasons"]

            st.session_state.stats["total"] += 1
            st.session_state.stats["passed"] += int(passes_filters)
            st.session_state.stats["rejected"] += int(not passes_filters)
            st.session_state.stats["similarities"].append(sim_score)

            st.subheader("Texte généré")
            st.write(generated)
            st.subheader("Résumé")
            st.write(summary)

            st.subheader("Qualité & Filtres")
            st.write(f"🔎 Similarité prompt/résumé : {sim_score:.2f} ({'OK' if is_relevant else 'BAS'})")
            st.write(f"🚡️ Filtrage éthique : {'✅ Accepté' if passes_filters else '❌ Rejeté'}")
            if not passes_filters:
                st.write("**Raisons du rejet :**")
                for r in reasons:
                    st.write(f"- {r}")

            with st.spinner("Calcul de perplexité..."):
                try:
                    res = requests.post(f"{API_URL}/perplexity", json={"prompt": generated})
                    ppl = res.json()["perplexity"]
                    st.write(f"🧹 Perplexité du texte généré : {ppl:.2f}")
                except Exception as e:
                    st.warning(f"Erreur perplexité : {e}")

        except Exception as e:
            st.error(f"Erreur d'appel à l'API : {e}")

st.sidebar.header("📊 Statistiques en temps réel")
st.sidebar.write(f"Total générés : {st.session_state.stats['total']}")
st.sidebar.write(f"✅ Acceptés : {st.session_state.stats['passed']}")
st.sidebar.write(f"❌ Rejetés : {st.session_state.stats['rejected']}")

if st.session_state.stats['similarities']:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.stats['similarities'], mode='lines+markers', name='Similarité'))
    fig.update_layout(title='Évolution de la similarité (prompt vs résumé)', yaxis=dict(range=[0, 1]))
    st.sidebar.plotly_chart(fig, use_container_width=True)
