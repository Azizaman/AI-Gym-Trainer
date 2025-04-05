import { useState } from 'react';
import { Brain, Activity, Zap } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { signInWithPopup } from "firebase/auth";
import { auth, provider } from "../utils/firebase";

function LandingPage() {
  const navigate = useNavigate();
  const [isContactModalOpen, setIsContactModalOpen] = useState(false);

  const openContactModal = () => setIsContactModalOpen(true);
  const closeContactModal = () => setIsContactModalOpen(false);
  const handleLogin = async () => {
      const result = await signInWithPopup(auth, provider);
      const token = await result.user.getIdToken();
  
      // ✅ Store token in localStorage
      localStorage.setItem("token", token);
  
      // ✅ Optionally send token to backend
      
  
      navigate("/choice"); // Redirect to the choice page after login
    };

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-800 text-white">
      {/* Header */}
      <header className="w-full px-4 py-6 flex justify-between items-center relative z-10 bg-opacity-90 backdrop-blur-sm">
        <Link>
          <div className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
            AI Fitness Assistant
          </div>
        </Link>

        <nav className="hidden md:flex space-x-6">
          <a href="#features" className="hover:text-blue-400 transition-all duration-200">Features</a>
          <a href="#about" className="hover:text-blue-400 transition-all duration-200">About</a>
          <button onClick={openContactModal} className="hover:text-blue-400 transition-all duration-200">
            Contact Us
          </button>
        </nav>

        <button className="hidden md:inline-flex px-5 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg shadow-lg hover:scale-105 transition-transform" onSubmit={handleLogin}>
          Sign Up
        </button>
      </header>

      {/* Main Content */}
      <main className="flex-grow w-full">
        <section className="relative w-full h-screen bg-cover bg-center bg-gradient-to-t from-gray-900 to-black">
          <div className="absolute inset-0 z-0">
            <img
              src="image1.jpg"
              alt="Fitness background"
              className="w-full h-full object-cover opacity-30"
            />
          </div>
          <div className="relative z-10 w-full px-6 py-32 text-center">
            <h2 className="text-4xl md:text-6xl font-bold mb-6 text-white">
              YOUR AI-POWERED <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">FITNESS JOURNEY</span> STARTS HERE
            </h2>
            <p className="text-2xl mb-10 max-w-3xl mx-auto text-gray-300">
              Personalized feedback, real-time form correction, and reps counter - all powered by cutting-edge AI technology.
            </p>
            <button onClick={handleLogin} className=" bg-gradient-to-r from-blue-400 to-purple-600 text-white px-8 py-4 rounded-lg shadow-lg text-lg font-bold hover:bg-cyan-600 transition-colors">
              Start Your Journey
            </button>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="bg-gray-900 py-20 w-full">
          <div className="w-full px-4">
            <h2 className="text-4xl font-extrabold text-center mb-12 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
              Why Choose AI Fitness Assistant?
            </h2>
            <div className="grid md:grid-cols-3 gap-10 max-w-6xl mx-auto">
              <FeatureCard
                icon={<Brain className="w-14 h-14 mb-6 text-blue-400" />}
                title="Exercise Feedback"
                description="Our AI analyzes your form and gives real-time feedback to prevent injuries."
              />
              <FeatureCard
                icon={<Activity className="w-14 h-14 mb-6 text-blue-400" />}
                title="AI Training"
                description="Receive AI-guided training to optimize your workout routine."
              />
              <FeatureCard
                icon={<Zap className="w-14 h-14 mb-6 text-blue-400" />}
                title="Reps Counter"
                description="Track reps automatically, including the incorrect ones, to improve your performance."
              />
            </div>
          </div>
        </section>

        {/* About Section */}
        <section id="about" className="w-full px-4 py-20 bg-gradient-to-r from-gray-900 to-black">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-4xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
              About AI Fitness Assistant
            </h2>
            <p className="text-lg mb-8 text-gray-400">
              Combining AI with fitness expertise to deliver real-time personalized feedback, helping you achieve your fitness goals efficiently and safely.
            </p>
            <button className="px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg shadow-lg text-lg font-semibold hover:scale-105 transition-transform" 
            onClick={() => navigate("choice")}
            >
              Learn More
            </button>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 py-8 w-full">
        <div className="w-full px-4 text-center text-gray-400">
          <p>&copy; {new Date().getFullYear()} AI Fitness. All rights reserved.</p>
          
        </div>
      </footer>

      {/* Contact Us Modal */}
      {isContactModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-gray-900 text-white p-8 rounded-lg shadow-lg max-w-lg w-full relative">
            <h2 className="text-3xl font-bold mb-4 text-center">
              Contact Us
            </h2>
            <form className="space-y-4">
              <input
                type="text"
                placeholder="Your Name"
                className="w-full px-4 py-2 bg-gray-800 text-white rounded-lg"
              />
              <input
                type="email"
                placeholder="Your Email"
                className="w-full px-4 py-2 bg-gray-800 text-white rounded-lg"
              />
              <textarea
                placeholder="Your Message"
                className="w-full px-4 py-2 bg-gray-800 text-white rounded-lg h-32"
              />
              <button
                type="submit"
                className="w-full px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg"
              >
                Send Message
              </button>
            </form>
            <button
              className="absolute top-2 right-2 text-white bg-red-500 hover:bg-red-600 p-2 rounded-full"
              onClick={closeContactModal}
            >
              X
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-8 rounded-lg shadow-lg hover:scale-105 transition-transform text-center">
      <div className="flex justify-center">{icon}</div>
      <h3 className="text-2xl font-semibold mb-4 text-white">{title}</h3>
      <p className="text-gray-400">{description}</p>
    </div>
  );
}

export default LandingPage;
