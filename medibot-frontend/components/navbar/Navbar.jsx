"use client";

import Image from "next/image";
import Link from "next/link";
import React, { useEffect, useState } from "react";

import logo from "../../public/assets/mediassist_logo.webp";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://localhost:8000";

const Navbar = () => {
  const [hasEvaluation, setHasEvaluation] = useState(false);

  useEffect(() => {
  const checkEvaluations = async () => {
    try {
      const response = await fetch(`${API_URL}/evaluations`, {
        cache: "no-store",
      });

      if (!response.ok) {
        setHasEvaluation(false);
        return;
      }

      const data = await response.json();

      console.log("Evaluations:", data);

      setHasEvaluation(
        Array.isArray(data.evaluations) &&
        data.evaluations.length > 0
      );
    } catch (error) {
      console.error("Failed to check evaluations:", error);
      setHasEvaluation(false);
    }
  };

  checkEvaluations();

  const interval = setInterval(checkEvaluations, 2000);

  return () => clearInterval(interval);
}, []);

  return (
    <nav className="flex w-full h-16 items-center justify-between px-4 border border-primary">
      <div className="flex items-center">
        <Link
          href="/"
          className="flex items-center"
        >
          <Image
            src={logo}
            alt="MediAssist"
            width={150}
            height={50}
            priority
          />
        </Link>
      </div>

      <div className="flex items-center gap-6 mr-10">
        <Link
          href="/"
          className="text-medium font-medium text-black transition-colors hover:text-blue-600"
        >
          Chat
        </Link>

        {hasEvaluation && (
          <Link
            href="/evaluations"
            className="text-medium font-medium text-black transition-colors hover:text-blue-600"
          >
            Evaluation
          </Link>
        )}
      </div>
    </nav>
  );
};

export default Navbar;