"use client";

import { useMotionValue, motion, useMotionTemplate } from "motion/react";
import React from "react";
import { cn } from "@/lib/utils";

export const CardSpotlight = ({
  children,
  radius = 350,
  color = "#4f378a",
  className,
  ...props
}) => {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  function handleMouseMove({ currentTarget, clientX, clientY }) {
    const { left, top } = currentTarget.getBoundingClientRect();

    mouseX.set(clientX - left);
    mouseY.set(clientY - top);
  }

  return (
    <div
      className={cn(
        "group/spotlight relative overflow-hidden rounded-md border border-neutral-200 bg-white p-4 text-black",
        className
      )}
      onMouseMove={handleMouseMove}
      {...props}
    >
      {/* Smooth circular spotlight */}
      <motion.div
        className="pointer-events-none absolute -inset-px rounded-md opacity-0 transition-opacity duration-300 group-hover/spotlight:opacity-100"
        style={{
          background: useMotionTemplate`
            radial-gradient(
              ${radius}px circle at ${mouseX}px ${mouseY}px,
              ${color}80,
              ${color}18 30%,
              transparent 70%
            )
          `,
        }}
      />

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};