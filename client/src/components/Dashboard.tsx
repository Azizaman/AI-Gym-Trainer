import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { PlusCircle } from "lucide-react";
import { Dropdown } from "./Dropdown";
import axios from "axios";
import { DialogTitle } from "@/components/ui/dialog";

type Video = {
  id: string;
  exercise: string;
  filename: string;
  video_url: string; // Processed video URL (signed)
  original_url?: string; // Original video Blob URL
  reps: number;
  feedback: string[];
  uploaded_at: number;
  duration: number;
};

export default function Dashboard() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [newTitle, setNewTitle] = useState("");
  const [newFile, setNewFile] = useState<File | null>(null);
  const [selectedExercise, setSelectedExercise] = useState<string>("");
  const [isUploading, setIsUploading] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch videos on mount
  useEffect(() => {
    const fetchVideos = async () => {
      setIsLoading(true);
      try {
        const token = localStorage.getItem("token");
        if (!token) {
          throw new Error("No authentication token found");
        }
        const response = await axios.get(`${import.meta.env.VITE_API_BASE_URL}/videos`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          params: {
            limit: 10,
            offset: 0,
          },
        });
        setVideos(response.data.videos);
        setIsLoading(false);
      } catch (err) {
        console.error("Error fetching videos:", err);
        setError("Failed to load videos. Please try again.");
        setIsLoading(false);
      }
    };

    fetchVideos();
  }, []);

  const handleVideoUpload = async () => {
    if (!newFile || !selectedExercise) {
      alert("Please select both a video file and an exercise type");
      return;
    }

    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", newFile);
      formData.append("exercise", selectedExercise);
      formData.append("title", newTitle || "Untitled Video");
      const token = localStorage.getItem("token");
      if (!token) {
        throw new Error("No authentication token found");
      }

      // Create a Blob URL for the original video
      const originalUrl = URL.createObjectURL(newFile);

      const response = await axios.post(`${import.meta.env.VITE_API_BASE_URL}/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.data) {
        const newVideo: Video = {
          id: response.data.id || Date.now().toString(),
          exercise: response.data.exercise,
          filename: newFile.name,
          video_url: response.data.video_url,
          original_url: originalUrl,
          reps: response.data.reps,
          feedback: response.data.feedback,
          uploaded_at: Date.now() / 1000,
          duration: response.data.duration,
        };

        setVideos((prev) => [newVideo, ...prev]);
        setNewFile(null);
        setNewTitle("");
        setSelectedExercise("");
      }
    } catch (error) {
      console.error("Error uploading video:", error);
      alert("Failed to upload video. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const openVideoModal = (video: Video) => {
    setSelectedVideo(video);
  };

  const closeVideoModal = () => {
    setSelectedVideo(null);
  };

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
          Your Fitness Videos
        </h1>

        <Dialog>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r cursor-pointer from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white border-none">
              <PlusCircle className="mr-2 h-5 w-5" />
              Upload New Video
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-gray-900 border border-gray-800">
            <h2 className="text-xl font-semibold mb-4 text-white">Upload New Video</h2>

            <div className="mb-4">
              <Dropdown onValueChange={setSelectedExercise} />
            </div>

            <Input
              type="file"
              accept="video/*"
              onChange={(e) => setNewFile(e.target.files?.[0] || null)}
              className="mb-4 bg-gray-800 border-gray-700 text-white"
            />
            <Input
              type="text"
              placeholder="Video Title (optional)"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.target)}
              className="mb-4 bg-gray-800 border-gray-700 text-white"
            />
            <Button
              onClick={handleVideoUpload}
              disabled={!newFile || !selectedExercise || isUploading}
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-800 hover:to-pink-700 text-white"
            >
              {isUploading ? "Uploading..." : "Upload"}
            </Button>
          </DialogContent>
        </Dialog>
      </div>

      {isLoading && <p className="text-white">Loading videos...</p>}
      {error && <p className="text-red-500">{error}</p>}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
        {videos.map((video) => (
          <Card
            key={video.id}
            className="transform hover:scale-105 transition-all duration-300 bg-gray-900 border border-gray-800 hover:border-purple-500 cursor-pointer"
            onClick={() => openVideoModal(video)}
          >
            <CardContent className="p-0">
              <video
                src={video.video_url}
                muted
                className="w-full h-48 object-cover rounded-t-lg"
              />
              <div className="p-6">
                <CardTitle className="text-xl font-semibold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
                  {video.filename}
                </CardTitle>
                <p className="text-sm text-gray-400">Exercise: {video.exercise.replace("_", " ").toUpperCase()}</p>
                <p className="text-sm text-gray-400">Reps: {video.reps}</p>
                <p className="text-sm text-gray-400">Duration: {video.duration.toFixed(2)}s</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {selectedVideo && (
        <Dialog open={!!selectedVideo} onOpenChange={closeVideoModal}>
          <DialogContent className="bg-gray-900 border border-gray-800 max-w-4xl">
            <DialogTitle className="text-xl font-semibold text-white">
              {selectedVideo.filename}
            </DialogTitle>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Processed Video</h3>
                <video
                  src={selectedVideo.video_url}
                  controls
                  className="w-full h-64 object-cover rounded-lg"
                />
              </div>
              {selectedVideo.original_url && (
                <div>
                  <h3 className="text-lg font-semibold text-white mb-2">Original Video</h3>
                  <video
                    src={selectedVideo.original_url}
                    controls
                    className="w-full h-64 object-cover rounded-lg"
                  />
                </div>
              )}
            </div>
            <div className="mt-4">
              <h3 className="text-lg font-semibold text-white">Details</h3>
              <p className="text-sm text-gray-400">Exercise: {selectedVideo.exercise.replace("_", " ").toUpperCase()}</p>
              <p className="text-sm text-gray-400">Reps: {selectedVideo.reps}</p>
              <p className="text-sm text-gray-400">Duration: {selectedVideo.duration.toFixed(2)}s</p>
              <p className="text-sm text-gray-400">Feedback:</p>
              <ul className="list-disc pl-5 text-sm text-gray-400">
                {selectedVideo.feedback.map((fb, index) => (
                  <li key={index}>{fb}</li>
                ))}
              </ul>
              <p className="text-sm text-gray-400">
                Uploaded: {new Date(selectedVideo.uploaded_at * 1000).toLocaleString()}
              </p>
            </div>
            <Button
              onClick={closeVideoModal}
              className="mt-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-800 hover:to-pink-700 text-white"
            >
              Close
            </Button>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}