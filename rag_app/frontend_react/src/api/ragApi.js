/**
 * ragApi.js — Typed API client for the RAG FastAPI backend.
 */

const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api/v1";

async function request(path, options = {}) {
  const url = `${BASE}${path}`;
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Send a chat message.
 * @param {string} question
 * @param {string} sessionId
 * @param {number} topK
 * @param {number} temperature
 */
export async function chatApi(question, sessionId, topK = 5, temperature = 0.2) {
  return request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId, top_k: topK, temperature }),
  });
}

/**
 * Upload a PDF file.
 * @param {File} file
 */
export async function uploadApi(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/upload", { method: "POST", body: form });
}

/**
 * List indexed document filenames.
 * @returns {Promise<string[]>}
 */
export async function documentsApi() {
  const data = await request("/documents");
  return data.documents || [];
}

/**
 * Rebuild the FAISS index from all uploaded PDFs.
 */
export async function rebuildApi() {
  return request("/rebuild", { method: "POST" });
}

/**
 * Fetch analytics summary.
 */
export async function analyticsApi() {
  return request("/analytics");
}
