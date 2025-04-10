import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { PlusCircle } from "lucide-react";

type Video = {
  id: string;
  title: string;
  url: string; // Video URL (processed)
  thumbnail: string;
};

const mockVideos: Video[] = [
  {
    id: "1",
    title: "Chest Day Form",
    url: "https://example.com/video1.mp4",
    thumbnail: "https://via.placeholder.com/300x200.png?text=Chest+Video",
  },
  {
    id: "2",
    title: "Leg Day Review",
    url: "https://example.com/video2.mp4",
    thumbnail: "https://via.placeholder.com/300x200.png?text=Leg+Video",
  },
];

export default function Dashboard() {
  const [videos, setVideos] = useState<Video[]>(mockVideos);
  const [newTitle, setNewTitle] = useState("");
  const [newFile, setNewFile] = useState<File | null>(null);

  const handleVideoUpload = () => {
    if (newFile) {
      // Here you should upload the file to Firebase Storage & trigger backend processing
      const fakeVideo: Video = {
        id: Date.now().toString(),
        title: newTitle || "Untitled Video",
        url: URL.createObjectURL(newFile), // Temporary URL
        thumbnail: "https://via.placeholder.com/300x200.png?text=New+Video",
      };

      setVideos((prev) => [fakeVideo, ...prev]);
      setNewFile(null);
      setNewTitle("");
    }
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
            <Input
              type="text"
              placeholder="Enter a title"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              className="mb-4 bg-gray-800 border-gray-700 text-white"
            />
            <Input
              type="file"
              accept="video/*"
              onChange={(e) => setNewFile(e.target.files?.[0] || null)}
              className="mb-4 bg-gray-800 border-gray-700 text-white"
            />
            <Button 
              onClick={handleVideoUpload} 
              disabled={!newFile}
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white"
            >
              Upload
            </Button>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
        {videos.map((video) => (
          <Card 
            key={video.id} 
            className="transform hover:scale-105 transition-all duration-300 bg-gray-900 border border-gray-800 hover:border-purple-500"
          >
            <CardContent className="p-0">
              <video
                src={video.url}
                controls
                className="w-full h-48 object-cover rounded-t-lg"
              />
              <div className="p-6">
                <CardTitle className="text-xl font-semibold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
                  {video.title}
                </CardTitle>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
