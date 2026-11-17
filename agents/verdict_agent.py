from backend.models.evaluation import JudgeResult


class VerdictAgent:
    WEIGHTS = {
        "relevance": 0.20,
        "accuracy": 0.30,
        "faithfulness": 0.30,
        "completeness": 0.20,
    }

    def evaluate(self, results: dict[str, JudgeResult]) -> dict:
        available_results = {
            name: result
            for name, result in results.items()
            if name in self.WEIGHTS and result.score is not None
        }

        if not available_results:
            return {
                "overall_score": 0.0,
                "verdict": "INSUFFICIENT_EVIDENCE",
                "summary": "No evaluation dimensions had sufficient evidence.",
            }

        available_weight = sum(
            self.WEIGHTS[name] for name in available_results
        )

        overall_score = sum(
            result.score * self.WEIGHTS[name]
            for name, result in available_results.items()
        ) / available_weight

        overall_score = round(overall_score, 2)

        if overall_score >= 80:
            verdict = "PASS"
        elif overall_score >= 60:
            verdict = "REVIEW"
        else:
            verdict = "FAIL"

        unavailable = [
            name
            for name in self.WEIGHTS
            if name not in available_results
        ]

        summary = (
            f"Evaluation completed using "
            f"{len(available_results)} of 4 dimensions. "
            f"Overall score: {overall_score}/100."
        )

        if unavailable:
            summary += (
                " Insufficient evidence for: "
                + ", ".join(unavailable)
                + "."
            )

        return {
            "overall_score": overall_score,
            "verdict": verdict,
            "summary": summary,
        }
