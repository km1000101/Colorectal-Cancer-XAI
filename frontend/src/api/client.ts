export type ExplanationType = "gradcam" | "lime" | "shap";

export interface ModelScore {
  model_name: string;
  probabilities: Record<string, number>;
}

export interface PredictionResult {
  predicted_class: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  per_model_scores: ModelScore[];
}

export interface ExplainResult {
  prediction: PredictionResult;
  gradcam?: { heatmap_base64: string } | null;
  lime?: { overlay_base64: string } | null;
  shap?: { heatmap_base64: string } | null;
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export async function uploadAndExplain(
  file: File,
  modelName: string,
  explanationTypes: ExplanationType[]
): Promise<ExplainResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", modelName);
  explanationTypes.forEach((t) => form.append("explanation_types", t));

  const res = await fetch(`${API_BASE}/api/explain`, {
    method: "POST",
    body: form
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return (await res.json()) as ExplainResult;
}
