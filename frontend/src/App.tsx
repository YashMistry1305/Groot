import { useEffect, useRef, useState } from 'react'
import {
  Bot, FileImage, Leaf, Paperclip, Send, Sparkles, Sprout, X,
} from 'lucide-react'
import { Button } from './components/ui/button'
import './App.css'

type Source = { title: string; source: string; page?: number | null; text: string }
type Message = { role: 'user' | 'assistant'; content: string; imageUrl?: string; observations?: Record<string, unknown> | null; sources?: Source[] }
type Session = { id: string; title: string; messages: Message[] }

const apiBase = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:5000/api'
const starterSessions: Session[] = [
  { id: 'today-1', title: 'Early blight on heirloom tomatoes', messages: [] },
  { id: 'today-2', title: 'Powdery mildew in greenhouse cucumbers', messages: [] },
  { id: 'yesterday-1', title: 'Nitrogen deficiency vs iron chlorosis', messages: [] },
  { id: 'yesterday-2', title: 'Companion planting for sweet corn', messages: [] },
  { id: 'week-1', title: 'Apple scab treatment schedule', messages: [] },
  { id: 'week-2', title: 'Aphid biological control with ladybugs', messages: [] },
]

function App() {
  const [sessions, setSessions] = useState<Session[]>(starterSessions)
  const [activeId] = useState(starterSessions[0].id)
  const [question, setQuestion] = useState('')
  const [image, setImage] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | undefined>()
  const [loading, setLoading] = useState(false)
  const [loadingStage, setLoadingStage] = useState('Rooting')
  const [error, setError] = useState('')
  const [showSources, setShowSources] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const active = sessions.find((session) => session.id === activeId) ?? sessions[0]

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl)
    }
  }, [imageUrl])
  useEffect(() => {
    if (!loading) return
    const stages = ['Rooting', 'Stemming', 'Branching', 'Sprouting', 'Shooting', 'Budding', 'Blooming', 'Pollinating']
    let index = 0
    setLoadingStage(stages[index])
    const interval = window.setInterval(() => {
      index = (index + 1) % stages.length
      setLoadingStage(stages[index])
    }, 850)
    return () => window.clearInterval(interval)
  }, [loading])
  function updateActive(update: (session: Session) => Session) {
    setSessions((current) => current.map((session) => session.id === activeId ? update(session) : session))
  }
  function handleImage(next: File | undefined) {
    if (!next) return
    if (imageUrl) URL.revokeObjectURL(imageUrl)
    setImage(next); setImageUrl(URL.createObjectURL(next))
  }
  async function submitQuestion() {
    if (!question.trim() || loading) return
    const asked = question.trim(); const askedImageUrl = imageUrl
    const history = active.messages.map(({ role, content }) => ({ role, content }))
    updateActive((session) => ({ ...session, title: session.title === 'New diagnosis' ? asked.slice(0, 42) : session.title, messages: [...session.messages, { role: 'user', content: asked, imageUrl: askedImageUrl }] }))
    setQuestion(''); setLoading(true); setError('')
    try {
      const body = new FormData(); body.append('question', asked); body.append('history', JSON.stringify(history)); if (image) body.append('image', image)
      const response = await fetch(`${apiBase}/chat`, { method: 'POST', body }); const result = await response.json()
      if (!response.ok) throw new Error(result.error ?? 'The local Groot server could not answer.')
      updateActive((session) => ({ ...session, messages: [...session.messages, { role: 'assistant', content: result.answer, observations: result.observations, sources: result.sources }] }))
      setImage(null); setImageUrl(undefined)
    } catch (requestError) { setError(requestError instanceof Error ? requestError.message : 'Something went wrong.') } finally { setLoading(false) }
  }
  function handleComposerKey(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void submitQuestion() }
  }
  function handleComposerPaste(event: React.ClipboardEvent<HTMLTextAreaElement>) {
    const pastedImage = Array.from(event.clipboardData.items).find((item) => item.type.startsWith('image/'))?.getAsFile()
    if (pastedImage) {
      event.preventDefault()
      handleImage(pastedImage)
    }
  }

  return <div className="app-shell">
    <main className="workspace">
      <header className="topbar"><div className="topbar-brand"><div className="mini-mark"><Sprout size={14} /></div><div><strong>Groot <em>● Ready to help</em></strong><small>See. Understand. Grow.</small></div></div><div className="topbar-actions"><Button variant="outline"><FileImage size={14} /> Export Log</Button></div></header>
      <section className="chat-area">{active.messages.length === 0 ? <div className="empty-state"><div className="empty-icon"><Bot size={24} /></div><h1>What are we looking at today?</h1><p>Upload a plant photo or describe a symptom to begin.</p><div className="suggestions"><button onClick={() => setQuestion('My tomato leaves have brown circular spots. What could be wrong?')}>Brown spots on tomato leaves</button><button onClick={() => setQuestion('How can I tell whether my crop needs more nitrogen?')}>Check nutrient deficiency</button></div></div> : active.messages.map((message, index) => <MessageView key={`${active.id}-${index}`} message={message} onSources={() => setShowSources(true)} />)}{loading && <div className="typing"><div className="assistant-avatar"><Sprout size={15} /></div><strong>{loadingStage}</strong><span>Groot is growing your answer</span><i /><i /><i /></div>}{error && <div className="error-banner">{error}</div>}</section>
      {showSources && <SourcesPanel messages={active.messages} onClose={() => setShowSources(false)} />}
      <section className="composer-wrap"><div className="composer">{imageUrl && <div className="attachment"><img src={imageUrl} alt="Selected plant" /><span>{image?.name ?? 'Pasted image'}</span><button onClick={() => { setImage(null); setImageUrl(undefined) }} aria-label="Remove image"><X size={13} /></button></div>}<textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={handleComposerKey} onPaste={handleComposerPaste} placeholder="Ask Groot about your plants, diseases, or soil care..." rows={2} /><div className="composer-tools"><button onClick={() => fileInput.current?.click()}><Paperclip size={15} /> Add leaf photo</button><input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={(event) => handleImage(event.target.files?.[0])} /><button><Sparkles size={14} /> Analyze crop</button><span className="mic">◉</span><button className="send-button" onClick={() => void submitQuestion()} disabled={!question.trim() || loading} aria-label="Send message"><Send size={16} /></button></div></div><div className="disclaimer">Paste an image directly into the message box or add a leaf photo. Groot can make mistakes; verify critical decisions with a qualified agronomist.</div></section>
    </main>
  </div>
}

function MessageView({ message, onSources }: { message: Message; onSources: () => void }) {
  if (message.role === 'user') return <div className="user-message">{message.imageUrl && <img src={message.imageUrl} alt="Uploaded plant" />}<div className="user-bubble">{message.content}</div><small>10:42 AM</small></div>
  return <div className="assistant-message"><div className="assistant-meta"><div className="assistant-avatar"><Sprout size={15} /></div><strong>Groot</strong><span>Botanical Diagnostics</span><small>Just now</small></div><article className="answer-card"><div className="answer-copy">{message.content.split('\n').map((line, index) => <p key={index} className={line.endsWith(':') ? 'answer-heading' : ''}>{line || <br />}</p>)}</div>{message.observations && <div className="diagnosis-card"><div className="diagnosis-icon"><Leaf size={16} /></div><div><strong>{String(message.observations.plant ?? 'Plant health observation')}</strong><small>{String(message.observations.overall_observation ?? 'Visual observations ready for review')}</small></div><span>● High confidence</span></div>}{message.sources?.length ? <button className="sources-link" onClick={onSources}>View {message.sources.length} grounded sources →</button> : null}</article></div>
}

function SourcesPanel({ messages, onClose }: { messages: Message[]; onClose: () => void }) {
  const sources = messages.flatMap((message) => message.sources ?? [])
  return <div className="sources-panel"><div className="sources-head"><div><span className="eyebrow">GROUNDED RETRIEVAL</span><h2>Sources used by Groot</h2></div><button className="icon-button" onClick={onClose}><X size={18} /></button></div>{sources.length ? sources.map((source, index) => <div className="source-item" key={`${source.source}-${index}`}><strong>{source.title}</strong><small>{source.source}{source.page ? ` · page ${source.page}` : ''}</small><p>{source.text.slice(0, 180)}…</p></div>) : <p>No sources are available yet.</p>}</div>
}

export default App
