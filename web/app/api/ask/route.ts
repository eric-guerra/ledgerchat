// Server-side proxy: the browser never talks to the RAG API directly, and entity access is resolved
// on the server. Nothing the browser sends (headers, body) can widen which entities are searched.
import { NextRequest, NextResponse } from 'next/server';

const API = process.env.LEDGERCHAT_API ?? 'http://localhost:8000';

export async function POST(req: NextRequest) {
  const { question } = await req.json();
  if (typeof question !== 'string' || question.trim().length < 3) return NextResponse.json({ error: 'question required' }, { status: 400 });
  const entities = await entitiesForSession(req);
  if (!entities) return NextResponse.json({ error: 'not signed in' }, { status: 401 });
  const r = await fetch(`${API}/ask`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-entities': entities },
    body: JSON.stringify({ question }),
    cache: 'no-store',
  });
  return NextResponse.json(await r.json(), { status: r.status });
}

/** Production: look up the SSO session (cookie → IdP groups → entity list). Dev: DEV_ENTITIES. */
async function entitiesForSession(_req: NextRequest): Promise<string | null> {
  if (process.env.NODE_ENV !== 'production') return process.env.DEV_ENTITIES ?? 'US01';
  return null; // wire to the identity provider before deploying
}
