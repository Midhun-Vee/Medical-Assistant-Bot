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
        const response = await fetch(
          `${API_URL}/evaluations`
        );

        if (!response.ok) {
          setHasEvaluation(false);
          return;
        }

        const data = await response.json();

        setHasEvaluation(
          Array.isArray(data.evaluations) &&
            data.evaluations.length > 0
        );
      } catch (error) {
        console.error(
          "Failed to check evaluations:",
          error
        );

        setHasEvaluation(false);
      }
    };

    checkEvaluations();

    const updateEvaluation = () => {
      checkEvaluations();
    };

    window.addEventListener(
      "evaluation-created",
      updateEvaluation
    );

    return () => {
      window.removeEventListener(
        "evaluation-created",
        updateEvaluation
      );
    };
  }, []);

  return (
    <nav className="flex items-center justify-between">
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

      <div className="flex items-center gap-6">
        <Link
          href="/"
          className="text-sm font-medium text-gray-700 transition-colors hover:text-blue-600"
        >
          Chat
        </Link>

        {hasEvaluation && (
          <Link
            href="/evaluations"
            className="text-sm font-medium text-gray-700 transition-colors hover:text-blue-600"
          >
            Evaluation
          </Link>
        )}
      </div>

      <div className="w-[150px]" />
    </nav>
  );
};

export default Navbar;