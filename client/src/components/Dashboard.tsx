
/* eslint-disable @typescript-eslint/no-explicit-any */
// Dashboard.tsx

import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PlusCircle, Loader2, CheckCircle, AlertCircle, MessageCircle } from "lucide-react";
import axios from "axios";

const Dropdown = ({ onValueChange }: { onValueChange: (value: string) => void }) => {
  const exercises = ["squat", "push_up", "bicep_curl", "plank", "jumping_jack"];
  return (
    <select
      onChange={(e) => onValueChange(e.target.value)}
      className="w-full p-2 sm:p-3 bg-gray-800/90 border border-gray-700 rounded-xl text-gray-200 focus:ring-2 focus:ring-purple-500 transition-all text-xs sm:text-sm"
    >
      <option value="">Select Exercise</option>
      {exercises.map((ex) => (
        <option key={ex} value={ex}>
          {ex.replace("_", " ").toUpperCase()}
        </option>
      ))}
    </select>
  );
};

type Video = {
  s3_key: string;
  exercise: string;
  filename: string;
  video_url: string;
  correct_reps: number;
  incorrect_reps: number;
  top_feedback: string[];
  uploaded_at: number;
  duration: number;
};

type ChatMessage = {
  sender: 'user' | 'bot';
  text: string;
  timestamp: number;
};

export default function Dashboard() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [newTitle, setNewTitle] = useState("");
  const [newFile, setNewFile] = useState<File | null>(null);
  const [selectedExercise, setSelectedExercise] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchVideos();
  }, []);

  useEffect(() => {
    // Scroll to the bottom of the chat container when new messages are added
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const fetchVideos = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const token = localStorage.getItem("token");
      if (!token) throw new Error("No authentication token found");

      const response = await axios.get(`${import.meta.env.VITE_API_BASE_URL || "https://ai-gym-assistant-6.onrender.com"}/videos`, {
        headers: { Authorization: `Bearer ${token}` },
        params: { limit: 10, offset: 0 },
      });

      const fetchedVideos = response.data.videos.map((v: any) => ({
        s3_key: v.s3_key,
        exercise: v.exercise,
        filename: v.filename,
        video_url: v.video_url,
        correct_reps: v.correct_reps,
        incorrect_reps: v.incorrect_reps,
        top_feedback: v.feedback,
        uploaded_at: v.uploaded_at,
        duration: v.duration,
      }));

      setVideos(fetchedVideos);
    } catch (err: any) {
      setError(err.message || "Failed to load videos. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleVideoUpload = async () => {
    if (!newFile || !selectedExercise) {
      setError("Please select a video file and an exercise type");
      return;
    }

    setIsUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", newFile);
      formData.append("exercise", selectedExercise);
      const token = localStorage.getItem("token");
      if (!token) throw new Error("No authentication token found");

      const response = await axios.post(`${import.meta.env.VITE_API_BASE_URL || "https://ai-gym-assistant-10.onrender.com"}/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.data) {
        const newVideo: Video = {
          s3_key: response.data.s3_key || `temp-${Date.now()}`,
          exercise: response.data.exercise,
          filename: newFile.name,
          video_url: response.data.video_url,
          correct_reps: response.data.correct_reps,
          incorrect_reps: response.data.incorrect_reps,
          top_feedback: response.data.top_feedback,
          uploaded_at: Date.now() / 1000,
          duration: response.data.duration,
        };

        setVideos((prev) => [newVideo, ...prev]);
        setNewFile(null);
        setNewTitle("");
        setSelectedExercise("");
      }
    } catch (error: any) {
      setError(error.message || "Failed to upload video. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleChatSubmit = async () => {
    if (!chatInput.trim()) return;

    const userMessage: ChatMessage = {
      sender: 'user',
      text: chatInput,
      timestamp: Date.now() / 1000,
    };
    setChatMessages((prev) => [...prev, userMessage]);
    setChatInput("");
    setIsChatLoading(true);

    try {
      const token = localStorage.getItem("token");
      if (!token) throw new Error("No authentication token found");

      const response = await axios.post(
        `${import.meta.env.VITE_API_BASE_URL || "https://ai-gym-assistant-10.onrender.com"}/chat`,
        { message: chatInput },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      const botMessage: ChatMessage = {
        sender: 'bot',
        text: response.data.response,
        timestamp: response.data.timestamp,
      };
      setChatMessages((prev) => [...prev, botMessage]);
    } catch (error: any) {
      const errorMessage: ChatMessage = {
        sender: 'bot',
        text: "Sorry, I couldn't process your request. Please try again.",
        timestamp: Date.now() / 1000,
      };
      setChatMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const openVideoModal = (video: Video) => setSelectedVideo(video);
  const closeVideoModal = () => setSelectedVideo(null);
  const isPositiveFeedback = (feedback: string) => feedback.toLowerCase().includes("correct form");

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 to-black text-white flex flex-col items-center justify-start w-full min-w-full p-4 sm:p-6 lg:p-8 font-sans">
      <div className="w-full min-w-full max-w-7xl mx-auto flex flex-col items-center">
        <div className="w-full flex flex-col sm:flex-row items-center justify-between mb-6 sm:mb-8">
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent animate-fade-in tracking-tight drop-shadow-lg text-center sm:text-left">
            Your Fitness Journey
          </h1>

          <div className="flex items-center gap-4 mt-4 sm:mt-0">
            <Dialog>
              <DialogTrigger asChild>
                <Button className="bg-gradient-to-r from-purple-500 to-pink-500 hover:scale-105 hover:from-purple-600 hover:to-pink-600 transition-transform duration-300 text-white font-semibold px-4 sm:px-6 py-2 sm:py-3 rounded-xl shadow-md">
                  <PlusCircle className="mr-2 h-4 sm:h-5 w-4 sm:w-5" />
                  Upload New Video
                </Button>
              </DialogTrigger>
              <DialogContent className="bg-gray-900/95 backdrop-blur-md border border-gray-700 rounded-xl w-full sm:max-w-2xl p-4 sm:p-6 shadow-2xl mx-auto">
                <DialogHeader>
                  <DialogTitle className="text-lg sm:text-xl font-bold text-white bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
                    Upload New Video
                  </DialogTitle>
                </DialogHeader>
                <div className="mt-4 space-y-4">
                  <Dropdown onValueChange={setSelectedExercise} />
                  <Input type="file" accept="video/*" onChange={(e) => setNewFile(e.target.files?.[0] || null)} className="text-xs sm:text-sm" />
                  <Input
                    type="text"
                    placeholder="Video Title (optional)"
                    value={newTitle}
                    onChange={(e) => setNewTitle(e.target.value)}
                    className="text-xs sm:text-sm"
                  />
                  <Button
                    onClick={handleVideoUpload}
                    disabled={!newFile || !selectedExercise || isUploading}
                    className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-700 text-white font-semibold py-2 sm:py-3 rounded-xl shadow-lg transition duration-300 text-xs sm:text-sm"
                  >
                    {isUploading ? <Loader2 className="animate-spin mr-2 h-4 sm:h-5 w-4 sm:w-5" /> : <PlusCircle className="mr-2 h-4 sm:h-5 w-4 sm:w-5" />}
                    {isUploading ? "Processing..." : "Upload"}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
            <Button
              onClick={() => setIsChatOpen(!isChatOpen)}
              className="bg-gradient-to-r from-blue-500 to-teal-500 hover:scale-105 hover:from-blue-600 hover:to-teal-600 transition-transform duration-300 text-white font-semibold px-4 sm:px-6 py-2 sm:py-3 rounded-xl shadow-md"
            >
              <MessageCircle className="mr-2 h-4 sm:h-5 w-4 sm:w-5" />
              {isChatOpen ? "Close Chat" : "Chat with Coach"}
            </Button>
          </div>
        </div>

        {isChatOpen && (
          <div className="w-full max-w-2xl bg-gray-900/95 border border-gray-700 rounded-xl p-4 sm:p-6 mb-6 shadow-2xl mx-auto">
            <div className="flex flex-col h-80 sm:h-96">
              <div
                ref={chatContainerRef}
                className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-900"
              >
                {chatMessages.length === 0 ? (
                  <p className="text-gray-400 text-center text-sm">Ask your fitness coach for exercise or diet tips!</p>
                ) : (
                  chatMessages.map((msg, index) => (
                    <div
                      key={index}
                      className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[70%] p-3 rounded-lg ${
                          msg.sender === 'user'
                            ? 'bg-purple-500 text-white'
                            : 'bg-gray-800 text-gray-200'
                        }`}
                      >
                        <p className="text-xs sm:text-sm">{msg.text}</p>
                        <p className="text-xs text-gray-400 mt-1">
                          {new Date(msg.timestamp * 1000).toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  ))
                )}
              </div>
              <div className="flex items-center gap-2 mt-4">
                <Input
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask for workout or diet advice..."
                  className="flex-1 text-xs sm:text-sm bg-gray-800 text-white border-gray-700"
                  onKeyPress={(e) => e.key === 'Enter' && handleChatSubmit()}
                />
                <Button
                  onClick={handleChatSubmit}
                  disabled={isChatLoading || !chatInput.trim()}
                  className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-4 py-2 rounded-xl"
                >
                  {isChatLoading ? <Loader2 className="animate-spin h-4 w-4" /> : "Send"}
                </Button>
              </div>
            </div>
          </div>
        )}

        {isLoading ? (
          <div className="flex justify-center items-center py-10 text-purple-400 w-full">
            <Loader2 className="h-6 sm:h-8 w-6 sm:w-8 animate-spin" />
            <span className="ml-3 text-base sm:text-lg">Loading videos...</span>
          </div>
        ) : error ? (
          <div className="p-4 bg-red-900/80 border border-red-600 rounded-lg flex items-center justify-between w-full mx-auto max-w-4xl">
            <span className="text-xs sm:text-sm">{error}</span>
            <Button onClick={fetchVideos} className="bg-red-600 hover:bg-red-700 text-white px-3 sm:px-4 py-1 sm:py-2 rounded-lg text-xs sm:text-sm">
              Retry
            </Button>
          </div>
        ) : videos.length === 0 ? (
          <p className="text-gray-400 text-center text-base sm:text-lg py-10 w-full">No videos uploaded yet. Start by uploading your first workout video!</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 w-full max-w-7xl mx-auto">
            {videos.map((video) => (
              <Card
                key={video.s3_key}
                className="bg-gray-800/40 border border-gray-700 rounded-2xl shadow-lg hover:shadow-purple-500/30 hover:scale-[1.015] transition-transform duration-300 cursor-pointer w-full mx-auto"
                onClick={() => openVideoModal(video)}
              >
                <CardHeader className="p-3 sm:p-4">
                  <CardTitle className="text-base sm:text-lg font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent truncate text-center">
                    {video.filename}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3 sm:p-4 space-y-2">
                  <video src={video.video_url} muted className="w-full h-32 sm:h-36 object-cover rounded-lg shadow-inner mx-auto" />
                  <div className="text-xs sm:text-sm text-gray-300 space-y-1 text-center">
                    <p><span className="text-gray-100 font-semibold">Exercise:</span> {video.exercise.replace("_", " ").toUpperCase()}</p>
                    <p><span className="text-gray-100 font-semibold"> Reps:</span> {video.correct_reps}</p>
                    {/* <p><span className="text-gray-100 font-semibold">Incorrect Reps:</span> {video.incorrect_reps}</p> */}
                    <p><span className="text-gray-100 font-semibold">Duration:</span> {video.duration.toFixed(2)}s</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {selectedVideo && (
          <Dialog open={!!selectedVideo} onOpenChange={closeVideoModal}>
            <DialogContent className="bg-gray-900 border border-gray-700 rounded-xl w-full sm:max-w-3xl max-h-[85vh] overflow-y-auto p-4 sm:p-6 shadow-2xl mx-auto">
              <DialogHeader>
                <DialogTitle className="text-lg sm:text-xl font-bold text-white text-center">{selectedVideo.filename}</DialogTitle>
              </DialogHeader>
              <div className="space-y-4 mt-4">
                <video src={selectedVideo.video_url} controls className="w-full max-h-64 sm:max-h-80 rounded-lg mx-auto" />
                <div className="space-y-1 text-xs sm:text-sm text-gray-300 text-center">
                  <p><span className="font-medium text-white">Exercise:</span> {selectedVideo.exercise.replace("_", " ").toUpperCase()}</p>
                  <p><span className="font-medium text-white"> Reps:</span> {selectedVideo.correct_reps}</p>
                  {/* <p><span className="font-medium text-white">Incorrect Reps:</span> {selectedVideo.incorrect_reps}</p> */}
                  <p><span className="font-medium text-white">Duration:</span> {selectedVideo.duration.toFixed(2)}s</p>
                  <p><span className="font-medium text-white">Uploaded:</span> {new Date(selectedVideo.uploaded_at * 1000).toLocaleString()}</p>
                </div>
                <div className="space-y-2">
                  <h4 className="text-white font-semibold text-sm sm:text-base text-center">Feedback</h4>
                  {selectedVideo.top_feedback.length > 0 ? (
                    selectedVideo.top_feedback.map((fb, i) => {
                      const isPositive = isPositiveFeedback(fb);
                      return (
                        <div key={i} className={`p-2 rounded-lg text-xs sm:text-sm flex items-start gap-2 ${isPositive ? "bg-green-900/30 text-green-200" : "bg-yellow-900/30 text-yellow-200"}`}>
                          {isPositive ? <CheckCircle className="w-4 h-4 text-green-400 mt-0.5" /> : <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5" />}
                          <span>{fb}</span>
                        </div>
                      );
                    })
                  ) : (
                    <p className="italic text-gray-400 text-xs sm:text-sm text-center">No feedback available. Try adjusting your form or camera angle for better analysis.</p>
                  )}
                </div>
                <Button onClick={closeVideoModal} className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold py-2 sm:py-3 rounded-xl shadow-lg mt-2 text-xs sm:text-sm">
                  Close
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        )}
      </div>
    </div>
  );
}
