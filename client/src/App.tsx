import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./App.css";
import LiveStream from "./pages/Live_Stream";
import UploadVideo from "./pages/Upload_video";
import Process from "@/components/Process";
import LandingPage from "@/pages/LandingPage";
import { ThreeDCardDemo } from "@/components/Cardchoice";
import Login from "./components/Login";
import Dashboard from "./components/Dashboard";

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path={"/"} element={<LandingPage />}></Route>
          <Route path={"/live"} element={<LiveStream />}></Route>
          <Route path={"/upload"} element={<UploadVideo />}></Route>
          <Route path="/process" element={<Process />}></Route>
          <Route path="/choice" element={<ThreeDCardDemo/>}></Route>
          <Route path="/login" element={<Login/>}></Route>
          <Route path="/dashboard" element={<Dashboard/>}></Route>
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;

