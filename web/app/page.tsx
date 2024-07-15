'use client';
import { useState } from 'react';

type Citation = { document_title: string; cited_text: string; chunk_id: string };
type Answer = { text: string; citations: Citation[]; grounded: boolean; stop_reason: string | null };
type Turn = { q: string; a?: Answer; error?: string };

export default function Page() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim() || busy) return;
    const question = q.trim();
    setTurns((t) => [...t, { q: question }]);
    setQ('');
    setBusy(true);
    try {
      const r = await fetch('/api/ask', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ question }) });
      const body = await r.json();
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? (r.ok ? { ...x, a: body } : { ...x, error: body.detail ?? body.error }) : x)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 820, margin: '0 auto', padding: 16 }}>
      <h1>LedgerChat</h1>
      <p style={{ color: '#6b7280' }}>Answers come only from your entity&apos;s policies, close memos and GL extracts, with citations.</p>
      {turns.map((t, i) => (
        <section key={i} style={{ background: '#fff', borderRadius: 8, padding: 12, margin: '12px 0' }}>
          <p><b>{t.q}</b></p>
          {t.error && <p style={{ color: '#c62828' }}>{t.error}</p>}
          {!t.a && !t.error && <p style={{ color: '#6b7280' }}>Searching documents…</p>}
          {t.a && (
            <>
              <p style={{ whiteSpace: 'pre-wrap' }}>{t.a.text}</p>
              {!t.a.grounded && <p style={{ color: '#945700', fontSize: 13 }}>Not supported by a cited source — verify before relying on it.</p>}
              {t.a.citations.length > 0 && (
                <ol style={{ fontSize: 13, color: '#374151' }}>
                  {t.a.citations.map((c, j) => <li key={j}><b>{c.document_title}</b>: “{c.cited_text}”</li>)}
                </ol>
              )}
            </>
          )}
        </section>
      ))}
      <form onSubmit={ask} style={{ display: 'flex', gap: 8 }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. What was accrued freight at August close?" style={{ flex: 1, padding: 10, fontSize: 15 }} />
        <button disabled={busy} style={{ padding: '10px 16px' }}>Ask</button>
      </form>
    </main>
  );
}
