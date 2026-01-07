"use client"

import type React from "react"
import { useState, useRef } from "react"

export default function DeepfakeDetector() {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<any>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const [isRecording, setIsRecording] = useState(false)
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile && selectedFile.type.startsWith("video/")) {
      setFile(selectedFile)
      setRecordedBlob(null)
      setResults(null)
      setErrorMsg(null)
    } else {
      setErrorMsg("Please select a valid video file.")
    }
  }

  const startWebcam = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    })

    mediaStreamRef.current = stream

    if (videoRef.current) {
      videoRef.current.srcObject = stream
      videoRef.current.muted = true
      videoRef.current.playsInline = true
      await videoRef.current.play()   // ✅ THIS LINE FIXES BLACK SCREEN
    }

    setIsRecording(true)
    chunksRef.current = []

    const mediaRecorder = new MediaRecorder(stream)
    mediaRecorderRef.current = mediaRecorder

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" })
      setRecordedBlob(blob)
      setFile(new File([blob], "webcam-video.webm", { type: "video/webm" }))

      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop())
      }

      setIsRecording(false)
    }

    mediaRecorder.start()
  } catch (error) {
    console.error("Webcam error:", error)
    setErrorMsg("Unable to access webcam. Please allow camera permission.")
  }
}

  const stopWebcam = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop()
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    setProgress(0)
    setErrorMsg(null)

    const formData = new FormData()
    formData.append("file", file)

    const progressInterval = window.setInterval(() => {
      setProgress((prev) => Math.min(prev + 10, 90))
    }, 500)

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Upload failed")
      }

      const data = await response.json()
      setProgress(100)
      setResults(data)
    } catch (error: any) {
      console.error("[v0] Error:", error)
      setErrorMsg(error?.message || "Something went wrong during analysis.")
    } finally {
      clearInterval(progressInterval)
      setUploading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b">
        <div className="mx-auto max-w-5xl px-4 py-6 flex items-center justify-between">
          <h1 className="text-xl font-semibold tracking-tight">Deepfake Detector</h1>
          <span className="text-sm text-muted-foreground">Simple HTML/CSS/JS/React UI</span>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8 space-y-8">
        {/* Upload Section */}
        <section className="rounded-lg border bg-card">
          <div className="p-5 border-b">
            <h2 className="text-lg font-medium">Upload Video for Analysis</h2>
          </div>
          <div className="p-5 space-y-5">
            {/* Upload Tab */}
            <div className="rounded-lg border-2 border-dashed p-6 text-center">
              <input type="file" accept="video/*" onChange={handleFileSelect} id="video-upload" className="hidden" />
              <label
                htmlFor="video-upload"
                className="inline-flex cursor-pointer items-center gap-3 rounded-md border px-4 py-2 text-sm font-medium bg-white border-gray-300 text-black hover:bg-gray-50 transition-colors"
              >
                <span>Select video file</span>
              </label>
              <div className="mt-3 text-sm text-muted-foreground">
                {file ? file.name : "Supports MP4, AVI, MOV formats"}
              </div>
            </div>

            <div className="rounded-lg border-2 border-dashed p-6 text-center space-y-3">
              <div className="text-sm font-medium">Or Record from Webcam</div>
              {!isRecording ? (
                <button
                  onClick={startWebcam}
                  className="inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium bg-blue-600 text-white hover:bg-blue-700 transition-colors"
                >
                  Start Recording
                </button>
              ) : (
                <button
                  onClick={stopWebcam}
                  className="inline-flex items-center gap-2 rounded-md border px-4 py-2 text-sm font-medium bg-red-600 text-white hover:bg-red-700 transition-colors"
                >
                  Stop Recording
                </button>
              )}
              {isRecording && (
                <video ref={videoRef} autoPlay playsInline className="w-full max-w-sm rounded-lg border bg-black" />
              )}
            </div>

            {file && (
              <div className="flex justify-center">
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="min-w-[200px] rounded-md bg-blue-600 text-white px-4 py-2 font-medium disabled:opacity-60 hover:bg-blue-700 transition-colors"
                >
                  {uploading ? "Analyzing..." : "Analyze Video"}
                </button>
              </div>
            )}

            {uploading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Processing video...</span>
                  <span>{progress}%</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                  <div className="h-full bg-primary transition-all" style={{ width: `${progress}%` }} />
                </div>
              </div>
            )}

            {errorMsg && (
              <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                {errorMsg}
              </div>
            )}
          </div>
        </section>

        {/* Results Section */}
        {results && (
          <section className="rounded-lg border bg-card">
            <div className="p-5 border-b">
              <h2 className="text-lg font-medium">Analysis Results</h2>
            </div>
            <div className="p-5 space-y-6">
              {/* Main Result */}
              <div className="text-center p-6 rounded-lg border">
                <div
                  className={`text-3xl font-bold mb-2 ${results.is_deepfake ? "text-destructive" : "text-green-600"}`}
                >
                  {results.is_deepfake ? "⚠️ FAKE VIDEO" : "✓ REAL VIDEO"}
                </div>
                <div className="text-lg font-semibold text-foreground">
                  Confidence: {(results.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-muted-foreground mt-1">
                  Deepfake Probability: {(results.deepfake_probability * 100).toFixed(1)}%
                </div>
              </div>

              {/* Sample Frames Visualization Section */}
              {results.sample_frames && (
                <div className="space-y-4 border rounded-lg p-4 bg-muted/30">
                  <div className="text-sm font-semibold">Sample Frame Comparison</div>
                  <div className="grid grid-cols-2 gap-4">
                    {results.sample_frames.real_frame && (
                      <div className="text-center">
                        <img
                          src={`data:image/jpeg;base64,${results.sample_frames.real_frame}`}
                          alt="Real frame example"
                          className="w-full h-auto rounded-lg border-2 border-green-600 bg-black"
                        />
                        <div className="mt-2 text-xs font-semibold text-green-600">✓ Real Frame</div>
                      </div>
                    )}
                    {results.sample_frames.fake_frame && (
                      <div className="text-center">
                        <img
                          src={`data:image/jpeg;base64,${results.sample_frames.fake_frame}`}
                          alt="Fake frame example"
                          className="w-full h-auto rounded-lg border-2 border-destructive bg-black"
                        />
                        <div className="mt-2 text-xs font-semibold text-destructive">⚠️ Fake Frame</div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Heatmap Visualization Section */}
              {results.heatmap && (
                <div className="space-y-4 border rounded-lg p-4 bg-muted/30">
                  <div className="text-sm font-semibold">Deepfake Detection Heatmap</div>
                  <div className="flex justify-center">
                    <img
                      src={`data:image/jpeg;base64,${results.heatmap}`}
                      alt="Deepfake detection heatmap showing suspicious regions"
                      className="max-w-lg w-full rounded-lg border"
                    />
                  </div>
                  <div className="text-xs text-muted-foreground text-center">
                    Red regions = High deepfake likelihood | Orange = Medium | Green = Natural
                  </div>
                </div>
              )}

              {/* Statistics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 rounded-lg border bg-muted/50">
                  <div className="text-2xl font-bold text-primary">{results.statistics.total_frames_analyzed}</div>
                  <div className="text-xs text-muted-foreground mt-1">Frames Analyzed</div>
                </div>
                <div className="text-center p-4 rounded-lg border bg-destructive/10">
                  <div className="text-2xl font-bold text-destructive">{results.statistics.fake_frames}</div>
                  <div className="text-xs text-muted-foreground mt-1">Fake Frames</div>
                </div>
                <div className="text-center p-4 rounded-lg border bg-green-50 dark:bg-green-950">
                  <div className="text-2xl font-bold text-green-600">{results.statistics.real_frames}</div>
                  <div className="text-xs text-muted-foreground mt-1">Real Frames</div>
                </div>
                <div className="text-center p-4 rounded-lg border bg-muted/50">
                  <div className="text-2xl font-bold text-primary">
                    {(results.statistics.average_confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">Avg Confidence</div>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-medium">
                  <span>Overall Confidence Score</span>
                  <span>{(results.statistics.average_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="h-3 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full bg-gradient-to-r from-green-500 to-primary transition-all"
                    style={{ width: `${results.statistics.average_confidence * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  )
}
