import React from 'react'
import { Button } from './ui/button'

export function JsonEditor({
  value,
  onChange,
}: {
  value: string
  onChange: (next: string) => void
}) {
  const validate = () => {
    try {
      JSON.parse(value)
      alert('Valid JSON ✅')
    } catch (e: any) {
      alert('Invalid JSON ❌\n\n' + e.message)
    }
  }

  const prettify = () => {
    try {
      const pretty = JSON.stringify(JSON.parse(value), null, 2)
      onChange(pretty)
    } catch {
      // ignore; user can validate first
    }
  }

  const minify = () => {
    try {
      const compact = JSON.stringify(JSON.parse(value))
      onChange(compact)
    } catch {
      // ignore
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <Button size="sm" onClick={validate}>
          <span title="Check JSON syntax">Validate</span>
        </Button>
        <Button size="sm" variant="outline" onClick={prettify}>
          <span title="Format with indentation">Prettify</span>
        </Button>
        <Button size="sm" variant="outline" onClick={minify}>
          <span title="Remove whitespace">Minify</span>
        </Button>
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full h-[420px] font-mono text-sm border rounded-xl p-3 outline-none"
        placeholder={`{
  "name": "",
  "type": "bacteria"
}`}
      />
    </div>
  )
}
