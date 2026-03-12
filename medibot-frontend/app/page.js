import Footer from "@/components/footer/Footer";
import Navbar from "@/components/navbar/Navbar";
import Home from "@/sections/Home";
import Image from "next/image";

export default function Page() {
  return (
  <div>
    <Navbar />
    <Home />
    <Footer />
  </div>
  );
}
