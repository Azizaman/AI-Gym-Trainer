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
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold">Your Fitness Videos</h1>

        <Dialog>
          <DialogTrigger asChild>
            <Button variant="default">
              <PlusCircle className="mr-2 h-5 w-5" />
              Upload New Video
            </Button>
          </DialogTrigger>
          <DialogContent>
            <h2 className="text-lg font-semibold mb-4">Upload New Video</h2>
            <Input
              type="text"
              placeholder="Enter a title"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              className="mb-4"
            />
            <Input
              type="file"
              accept="video/*"
              onChange={(e) => setNewFile(e.target.files?.[0] || null)}
              className="mb-4"
            />
            <Button onClick={handleVideoUpload} disabled={!newFile}>
              Upload
            </Button>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {videos.map((video) => (
          <Card key={video.id} className="hover:shadow-lg transition-shadow">
            <CardContent className="p-0">
              <video
                src={video.url}
                controls
                className="w-full h-48 object-cover rounded-t-lg"
              />
              <div className="p-4">
                <CardTitle className="text-lg">{video.title}</CardTitle>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
