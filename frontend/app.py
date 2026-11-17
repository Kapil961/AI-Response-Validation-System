
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/api/v1/submissions"

st.set_page_config(
    page_title="AI Response Validation System",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 AI Response Validation System")

st.markdown(
    """
    Evaluate AI-generated responses for:
    - Relevance
    - Accuracy
    - Faithfulness and hallucination detection
    - Completeness
    """
)

st.info(
    "Enter a question and its AI-generated response. "
    "Reference answers and source documents are optional."
)


# --------------------------------------------------
# 1. Submission Form
# --------------------------------------------------

with st.form("evaluation_form"):

    st.subheader("1. Submission Details")

    question = st.text_area(
        "Original Question *",
        placeholder="Enter the question asked by the user...",
        height=100,
    )

    ai_response = st.text_area(
        "AI-Generated Response *",
        placeholder="Paste the AI-generated answer here...",
        height=200,
    )

    reference_answer = st.text_area(
        "Reference Answer (Optional)",
        placeholder="Enter a trusted reference answer...",
        height=120,
    )

    source_document = st.text_area(
        "Source Document / Supporting Context (Optional)",
        placeholder="Paste the document or trusted context...",
        height=150,
    )

    submitted = st.form_submit_button(
        "Evaluate AI Response",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# 2. Input Validation and API Request
# --------------------------------------------------

if submitted:

    if not question.strip():
        st.error("Please enter the original question.")

    elif len(question.strip()) < 5:
        st.error("The question must contain at least 5 characters.")

    elif not ai_response.strip():
        st.error("Please enter the AI-generated response.")

    elif len(ai_response.strip()) < 10:
        st.error("The AI response must contain at least 10 characters.")

    else:

        payload = {
            "question": question.strip(),
            "ai_response": ai_response.strip(),
            "reference_answer": reference_answer.strip() or None,
            "source_document": source_document.strip() or None,
        }

        try:

            with st.spinner("Evaluating response... Please wait."):

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=60,
                )

            response.raise_for_status()

            data = response.json()
            result = data.get("result", {})

            st.success("Evaluation completed successfully!")

            # --------------------------------------------------
            # 3. Overall Evaluation
            # --------------------------------------------------

            st.divider()
            st.header("2. Evaluation Results")

            overall_score = result.get("overall_score", 0)
            verdict = result.get("verdict", "UNKNOWN")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Overall Score",
                    f"{overall_score:.2f}/100",
                )

            with col2:
                st.metric(
                    "Final Verdict",
                    verdict,
                )

            if verdict == "PASS":
                st.success("The response passed the evaluation.")

            elif verdict == "REVIEW":
                st.warning("The response requires further review.")

            elif verdict == "FAIL":
                st.error("The response failed the evaluation.")

            else:
                st.warning(
                    "Insufficient evidence for a reliable verdict."
                )

            st.subheader("Evaluation Summary")

            st.write(
                result.get(
                    "summary",
                    "No summary available.",
                )
            )

            # --------------------------------------------------
            # 4. Individual Judge Results
            # --------------------------------------------------

            st.divider()
            st.subheader("3. Individual Judge Results")

            judge_details = [
                ("Relevance", "relevance"),
                ("Accuracy", "accuracy"),
                ("Faithfulness / Hallucination", "faithfulness"),
                ("Completeness", "completeness"),
            ]

            columns = st.columns(2)

            for index, (label, key) in enumerate(judge_details):

                judge = result.get(key, {})

                score = judge.get("score")
                passed = judge.get("passed")
                status = judge.get("status", "unknown")

                with columns[index % 2]:

                    with st.container(border=True):

                        st.markdown(f"### {label}")

                        if score is not None:
                            st.metric(
                                "Score",
                                f"{score:.2f}/100",
                            )
                        else:
                            st.metric(
                                "Score",
                                "Not available",
                            )

                        if passed is True:
                            st.success("Passed")

                        elif passed is False:
                            st.error("Did not pass")

                        else:
                            st.warning("Not evaluated")

                        st.caption(f"Status: {status}")

                        st.write(
                            judge.get(
                                "explanation",
                                "No explanation available.",
                            )
                        )

                        evidence = judge.get("evidence")

                        if evidence:
                            with st.expander("View Evidence"):
                                st.write(evidence)

                        missing = judge.get("missing_concepts")

                        if missing:
                            with st.expander("Missing Concepts"):
                                st.write(missing)

                        covered = judge.get("covered_concepts")

                        if covered:
                            with st.expander("Covered Concepts"):
                                st.write(covered)

            # --------------------------------------------------
            # 5. Complete Evaluation JSON
            # --------------------------------------------------

            st.divider()

            with st.expander("View Complete Evaluation JSON"):
                st.json(result)

        # --------------------------------------------------
        # 6. Exception Handling
        # --------------------------------------------------

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the backend. "
                "Please start the FastAPI server on port 8000."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The evaluation request timed out. "
                "Please try again."
            )

        except requests.exceptions.HTTPError as error:

            st.error(
                f"Backend returned an HTTP error: {error}"
            )

            try:
                st.json(response.json())

            except ValueError:
                st.write(response.text)

        except requests.exceptions.RequestException as error:

            st.error(f"Request failed: {error}")

        except (ValueError, TypeError, KeyError) as error:

            st.error(
                f"Could not process the evaluation result: {error}"
            )


# --------------------------------------------------
# 7. Application Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI Response Validation System | "
    "Development of AI Response Validation System "
    "with Hallucination Detection Assistance"
)
