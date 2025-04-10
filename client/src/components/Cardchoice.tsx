import { CardBody, CardContainer, CardItem } from "@/components/ui/3d-card";
import { Link } from "react-router-dom";

export function ThreeDCardDemo() {
  return (
   <div>
    
    <header className="w-full px-4 py-5 flex justify-between items-center relative z-10 bg-opacity-90 backdrop-blur-sm bg-black">
        <Link to={'/'}>
          <div className="text-3xl font-extrabold text-transparent bg-clip-text bg-gray-300">
            
            AI Fitness Assistant
          </div>
        </Link>
       
      </header>


      <div className="bg-black min-h-screen flex gap-10 justify-center items-center py-10 ml-56">
        {/* Live Video Card */}
        <CardContainer className="inter-var">
          <CardBody
            className="bg-black relative group/card dark:hover:shadow-2xl dark:hover:shadow-emerald-500/[0.1] dark:bg-black border-white/[0.8] w-auto sm:w-[30rem] h-auto rounded-xl p-6 border border-[0.5px]"
          >
          <CardItem
            translateZ="50"
            className="text-xl font-bold text-neutral-600 dark:text-white"
          >
            Go Live
          </CardItem>
          <CardItem
            as="p"
            translateZ="60"
            className="text-neutral-500 text-sm max-w-sm mt-2 dark:text-neutral-300"
          >
            Start a live session and track your workout in real-time
          </CardItem>
          <CardItem translateZ="100" className="w-full mt-4">
            <img
              src="image5.jpg"
              height="1000"
              width="1000"
              className="h-60 w-full object-cover rounded-xl group-hover/card:shadow-xl"
              alt="Live session"
            />
          </CardItem>
          <div className="flex justify-between items-center mt-20">
            <CardItem
              translateZ={20}
              as={Link}
              to="/live"
              target="__blank"
              className="px-4 py-2 rounded-xl text-xl font-normal text-white"
            >
              Go Live →
            </CardItem>
          </div>
        </CardBody>
      </CardContainer>

      {/* Upload Video Card */}
      <CardContainer className="inter-var">
        <CardBody
          className="bg-black relative group/card dark:hover:shadow-2xl dark:hover:shadow-emerald-500/[0.1] dark:bg-black border-white/[0.8] w-auto sm:w-[30rem] h-auto rounded-xl p-6 border border-[0.5px]" // Thinner white border
        >
          <CardItem
            translateZ="50"
            className="text-xl font-bold text-neutral-600 dark:text-white"
          >
            Upload Video
          </CardItem>
          <CardItem
            as="p"
            translateZ="60"
            className="text-neutral-500 text-sm max-w-sm mt-2 dark:text-neutral-300"
          >
            Upload a pre-recorded workout video and analyze it.
          </CardItem>
          <CardItem translateZ="100" className="w-full mt-4">
            <img
              src="image4.jpg"
              height="1000"
              width="1000"
              className="h-60 w-full object-cover rounded-xl group-hover/card:shadow-xl"
              alt="Upload Video"
            />
          </CardItem>
          <div className="flex justify-between items-center mt-20">
            <CardItem
              translateZ={20}
              as={Link}
              to="/dashboard"
              target="__blank"
              className="px-4 py-2 rounded-xl text-xl font-normal text-white"
            >
              Upload Video →
            </CardItem>
          </div>
        </CardBody>
      </CardContainer>
    </div>
    </div>
  );
}
