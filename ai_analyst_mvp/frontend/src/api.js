const BASE_URL = "http://localhost:8000";

export const runEvaluation = async () => {
  const res = await fetch(`${BASE_URL}/evaluate`);
  if (!res.ok) {
    throw new Error("Evaluation failed");
  }
  return await res.json();
};

