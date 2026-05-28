import { useState, useRef, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import { Upload, Send, RefreshCw, Trash2, BookOpen, FileText, Loader2, ChevronDown, ChevronUp } from "lucide-react";
import { chatApi, uploadApi, documentsApi, rebuildApi, analyticsApi } from "./api/ragApi";

// ── Message bubble ────────────────────────────────────────────────────────────
function Message({ msg }) {
  const [showSources, setShowSources] = useState(false);
  const isUser = msg.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-[75%] ${isUser ? "order-2" : "order-1"}`}>
        <div
          className={`rounded-2xl px-4 py-3 shadow-sm ${
            isUser
              ? "bg-blue-600 text-white rounded-br-md"
              : "bg-white text-gray-800 rounded-bl-md border border-gray-100"
          }`}
        >
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
        </div>

        {!isUser && msg.latency_ms && (
          <p className="text-xs text-gray-400 mt-1 px-1">
            ⏱ {msg.latency_ms}ms · 📎 {msg.chunks_used} chunks
          </p>
        )}

        {!isUser && msg.sources?.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 px-1"
            >
              <BookOpen size={12} />
              {msg.sources.length} source{msg.sources.length !== 1 ? "s" : ""}
              {showSources ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>

            {showSources && (
              <div className="mt-2 space-y-2">
                {msg.sources.map((src, i) => (
                  <div key={i} className="bg-blue-50 border border-blue-100 rounded-lg p-3">
                    <p className="text-xs font-semibold text-blue-800">
                      {src.source} · Page {src.page}
                    </p>
                    <p className="text-xs text-gray-600 mt-1 italic">{src.snippet}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Sidebar({ onUpload, docs, onRefreshDocs, onRebuild, settings, onSettingsChange }) {
  const fileRef = useRef();
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg(null);
    try {
      const result = await uploadApi(file);
      setUploadMsg({ type: "success", text: `✅ ${result.file}: ${result.pages} pages, ${result.chunks} chunks` });
      onRefreshDocs();
    } catch (err) {
      setUploadMsg({ type: "error", text: `❌ Upload failed: ${err.message}` });
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  return (
    <aside className="w-72 bg-gray-50 border-r border-gray-200 flex flex-col h-full overflow-y-auto">
      <div className="p-4 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <BookOpen size={22} className="text-blue-600" /> RAG Chat
        </h1>
        <p className="text-xs text-gray-500 mt-1">Document Q&A powered by GPT</p>
      </div>

      {/* Settings */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Settings</h2>
        <label className="block text-xs text-gray-600 mb-1">Top-K: {settings.topK}</label>
        <input
          type="range" min={1} max={10} value={settings.topK}
          onChange={e => onSettingsChange({ ...settings, topK: +e.target.value })}
          className="w-full accent-blue-600"
        />
        <label className="block text-xs text-gray-600 mb-1 mt-3">Temperature: {settings.temperature}</label>
        <input
          type="range" min={0} max={1} step={0.05} value={settings.temperature}
          onChange={e => onSettingsChange({ ...settings, temperature: +e.target.value })}
          className="w-full accent-blue-600"
        />
      </div>

      {/* Upload */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Upload PDF</h2>
        <input type="file" accept=".pdf" ref={fileRef} onChange={handleFileChange} className="hidden" />
        <button
          onClick={() => fileRef.current?.click()}
          disabled={uploading}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm font-medium rounded-lg px-4 py-2 transition-colors"
        >
          {uploading ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
          {uploading ? "Uploading…" : "Choose PDF"}
        </button>
        {uploadMsg && (
          <p className={`text-xs mt-2 ${uploadMsg.type === "success" ? "text-green-600" : "text-red-500"}`}>
            {uploadMsg.text}
          </p>
        )}
      </div>

      {/* Documents */}
      <div className="p-4 border-b border-gray-200 flex-1">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Documents</h2>
          <button onClick={onRefreshDocs} className="text-gray-400 hover:text-gray-600">
            <RefreshCw size={12} />
          </button>
        </div>
        {docs.length === 0 ? (
          <p className="text-xs text-gray-400">No documents indexed yet.</p>
        ) : (
          <ul className="space-y-1">
            {docs.map((d, i) => (
              <li key={i} className="flex items-center gap-2 text-xs text-gray-700">
                <FileText size={12} className="text-blue-500 shrink-0" /> {d}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Actions */}
      <div className="p-4">
        <button
          onClick={onRebuild}
          className="w-full flex items-center justify-center gap-2 border border-gray-300 hover:bg-gray-100 text-gray-700 text-sm rounded-lg px-4 py-2 transition-colors"
        >
          <RefreshCw size={14} /> Rebuild Index
        </button>
      </div>
    </aside>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(() => uuidv4());
  const [docs, setDocs] = useState([]);
  const [settings, setSettings] = useState({ topK: 5, temperature: 0.2 });
  const bottomRef = useRef();

  useEffect(() => {
    fetchDocs();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchDocs = async () => {
    try {
      const list = await documentsApi();
      setDocs(list);
    } catch (_) {}
  };

  const handleRebuild = async () => {
    try {
      await rebuildApi();
      fetchDocs();
      alert("Index rebuilt successfully!");
    } catch (err) {
      alert(`Rebuild failed: ${err.message}`);
    }
  };

  const handleSend = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setInput("");
    setMessages(prev => [...prev, { role: "user", content: q }]);
    setLoading(true);

    try {
      const result = await chatApi(q, sessionId, settings.topK, settings.temperature);
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources || [],
          latency_ms: result.latency_ms,
          chunks_used: result.chunks_used,
        },
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: `❌ Error: ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      <Sidebar
        onUpload={() => {}}
        docs={docs}
        onRefreshDocs={fetchDocs}
        onRebuild={handleRebuild}
        settings={settings}
        onSettingsChange={setSettings}
      />

      {/* Chat Area */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Document Assistant</h2>
            <p className="text-xs text-gray-500">Session: {sessionId.slice(0, 8)}…</p>
          </div>
          <button
            onClick={() => { setMessages([]); }}
            className="flex items-center gap-2 text-sm text-gray-500 hover:text-red-500 transition-colors"
          >
            <Trash2 size={14} /> Clear
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <BookOpen size={48} className="text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500 text-lg font-medium">Ask about your documents</p>
                <p className="text-gray-400 text-sm mt-1">Upload PDFs in the sidebar, then start chatting</p>
              </div>
            </div>
          )}
          {messages.map((msg, i) => <Message key={i} msg={msg} />)}
          {loading && (
            <div className="flex justify-start mb-4">
              <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
                <Loader2 size={16} className="animate-spin text-blue-500" />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <div className="flex gap-3 items-end">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask something about your documents… (Enter to send)"
              rows={2}
              className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white rounded-xl p-3 transition-colors"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
