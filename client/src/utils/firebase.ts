// src/utils/firebase.ts

import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getStorage } from "firebase/storage";

// Your Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyCGJD3MdI2BJk3Oglm-vDxr96k7sr9nKic",
  authDomain: "ai-fitness-trainer-c65a8.firebaseapp.com",
  projectId: "ai-fitness-trainer-c65a8",
  storageBucket: "ai-fitness-trainer-c65a8.appspot.com", // fixed typo here!
  messagingSenderId: "217213173752",
  appId: "1:217213173752:web:1d45253603a72b05ec200d",
  measurementId: "G-1G486B0D83",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Optional: if you're using it
const analytics = getAnalytics(app);

// Setup Authentication
export const auth = getAuth(app);
export const provider = new GoogleAuthProvider();

// Setup Storage
export const storage = getStorage(app);