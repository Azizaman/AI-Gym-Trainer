// src/pages/Login.tsx
import { signInWithPopup } from "firebase/auth";
import { auth, provider } from "../utils/firebase";

import { useNavigate } from "react-router-dom";
const Login = () => {
  const navigate = useNavigate();
  const handleLogin = async () => {
    const result = await signInWithPopup(auth, provider);
    const token = await result.user.getIdToken();

    // ✅ Store token in localStorage
    localStorage.setItem("token", token);

    // ✅ Optionally send token to backend
    

    navigate("/choice"); // Redirect to the choice page after login
  };

  return (
    <button onClick={handleLogin} className="bg-blue-600 text-white px-4 py-2 rounded">
      Login with Google
    </button>
  );
};

export default Login;
