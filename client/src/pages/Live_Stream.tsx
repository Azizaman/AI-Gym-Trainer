import { useRef, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import { useNavigate } from 'react-router-dom';

const LiveStream = () => {
    const [downloadLink, setDownloadLink] = useState(null);
    const webcamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [recordedChunks, setRecordedChunks] = useState([]);
    const navigate = useNavigate();

    const handleStartCaptureClick = useCallback(() => {
        setRecordedChunks([]);
        mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
            mimeType: "video/webm",
        });
        mediaRecorderRef.current.addEventListener(
            "dataavailable",
            handleDataAvailable
        );
        mediaRecorderRef.current.start();
    }, [webcamRef, setRecordedChunks]);

    const handleDataAvailable = useCallback(({ data }) => {
        if (data.size > 0) {
            setRecordedChunks((prev) => prev.concat(data));
        }
    }, []);

    const handleStopCaptureClick = useCallback(() => {
        mediaRecorderRef.current.stop();
    }, [mediaRecorderRef]);

    const handleDownload = useCallback(() => {
        if (recordedChunks.length) {
            const blob = new Blob(recordedChunks, {
                type: "video/webm"
            });
            const url = URL.createObjectURL(blob);
            setDownloadLink(url);
        }
    }, [recordedChunks]);

    const redirectToUpload = () => {
        navigate('/upload');
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-6">
            <h1 className="text-4xl font-bold mb-8 text-center">AI Fitness Trainer For Gym</h1>
            
            <div className="bg-gray-800 p-8 rounded-lg shadow-lg flex flex-col items-center w-full max-w-2xl">
                <Webcam ref={webcamRef} className="rounded-lg mb-6 w-full" />

                <div className="flex space-x-4 mb-6">
                    <button 
                        onClick={handleStartCaptureClick} 
                        className="px-6 py-2 bg-blue-600 rounded-lg text-lg font-bold hover:bg-blue-500 transition"
                    >
                        Start Capture
                    </button>
                    <button 
                        onClick={handleStopCaptureClick} 
                        className="px-6 py-2 bg-red-600 rounded-lg text-lg font-bold hover:bg-red-500 transition"
                    >
                        Stop Capture
                    </button>
                    <button 
                        onClick={handleDownload} 
                        className="px-6 py-2 bg-green-600 rounded-lg text-lg font-bold hover:bg-green-500 transition"
                    >
                        Download Video
                    </button>
                    <button 
                        onClick={redirectToUpload} 
                        className="px-6 py-2 bg-purple-600 rounded-lg text-lg font-bold hover:bg-purple-500 transition"
                    >
                        Upload Video
                    </button>
                </div>

                {downloadLink && (
                    <a 
                        href={downloadLink} 
                        download="output_live.webm"
                        className="text-blue-400 underline hover:text-blue-500"
                    >
                        Download Captured Video
                    </a>
                )}
            </div>
        </div>
    );
};

export default LiveStream;
