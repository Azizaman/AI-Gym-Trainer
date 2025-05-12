import { CardBody, CardContainer, CardItem } from "@/components/ui/3d-card";
import { Link } from "react-router-dom";

export function ThreeDCardDemo() {
  return (
    <div className="bg-black min-h-screen w-full">
      {/* Header */}
      <header className="w-full px-4 sm:px-6 py-4 flex justify-between items-center bg-opacity-90 backdrop-blur-sm bg-black sticky top-0 z-10">
        <Link to={'/'}>
          <div className="text-2xl sm:text-3xl font-extrabold text-transparent bg-clip-text bg-gray-300">
            AI Fitness Assistant
          </div>
        </Link>
      </header>

      {/* Card Section */}
      <div className="flex flex-col items-center justify-center py-6 px-4 sm:px-6 w-full min-h-[calc(100vh-80px)]">
        {/* Upload Video Card */}
        <CardContainer className="inter-var w-full max-w-[30rem] mx-auto">
          <CardBody
            className="bg-black relative group/card dark:hover:shadow-2xl dark:hover:shadow-emerald-500/[0.1] dark:bg-black border-white/[0.8] w-full h-auto rounded-xl p-4 sm:p-6 border border-[0.5px]"
          >
            <CardItem
              translateZ="50"
              className="text-lg sm:text-xl font-bold text-neutral-600 dark:text-white"
            >
              Upload Video
            </CardItem>
            <CardItem
              as="p"
              translateZ="60"
              className="text-neutral-500 text-xs sm:text-sm max-w-sm mt-2 dark:text-neutral-300"
            >
              Upload a pre-recorded workout video and analyze it.
            </CardItem>
            <CardItem translateZ="100" className="w-full mt-4">
              <img
                src="image4.jpg"
                height="1000"
                width="1000"
                className="h-40 sm:h-60 w-full object-cover rounded-xl group-hover/card:shadow-xl"
                alt="Upload Video"
              />
            </CardItem>
            <div className="flex justify-between items-center mt-8 sm:mt-20">
              <CardItem
                translateZ={20}
                as={Link}
                to="/dashboard"
                target="_blank"
                className="px-3 sm:px-4 py-1 sm:py-2 rounded-xl text-base sm:text-xl font-normal text-white"
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