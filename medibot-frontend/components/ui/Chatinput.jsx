"use client";

import { useRef, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function ChatInput({
  onResponse,
  onNewQuery,
  chatId,
}) {
  const [message, setMessage] = useState("");
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);

  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const canSend =
    message.trim().length > 0 || files.length > 0;

  const handleMessageChange = (e) => {
    setMessage(e.target.value);

    const textarea = textareaRef.current;

    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(
        textarea.scrollHeight,
        176
      )}px`;
    }
  };

  const handleFiles = (e) => {
    if (!e.target.files) return;

    const selectedFiles = Array.from(e.target.files);

    const allowedTypes = [
      "application/pdf",
      "image/jpeg",
      "image/png",
      "image/webp",
    ];

    const validFiles = selectedFiles.filter((file) =>
      allowedTypes.includes(file.type)
    );

    if (validFiles.length !== selectedFiles.length) {
      window.alert(
        "Only PDF, JPG, JPEG, PNG, and WEBP files are supported."
      );
    }

    setFiles((current) => [
      ...current,
      ...validFiles,
    ]);

    e.target.value = "";
  };

  const removeFile = (index) => {
    setFiles((current) =>
      current.filter((_, i) => i !== index)
    );
  };

  const handleSubmit = async () => {
    if (!canSend || loading) return;

    const question = message.trim();

    setLoading(true);

    try {
      let currentChatId = chatId;

      if (!currentChatId) {
        currentChatId = crypto.randomUUID();
      }

      for (const file of files) {
        const formData = new FormData();

        formData.append("file", file);
        formData.append("chat_id", currentChatId);

        const uploadResponse = await fetch(
          `${API_URL}/upload`,
          {
            method: "POST",
            body: formData,
          }
        );

        const uploadResult =
          await uploadResponse.json();

        if (!uploadResponse.ok) {
          console.error(
            "Upload failed:",
            uploadResponse.status,
            uploadResult
          );

          let errorMessage =
            `Failed to upload ${file.name}`;

          if (
            typeof uploadResult.detail === "string"
          ) {
            errorMessage =
              uploadResult.detail;
          } else if (
            Array.isArray(uploadResult.detail)
          ) {
            errorMessage =
              uploadResult.detail
                .map((error) => {
                  if (
                    typeof error === "string"
                  ) {
                    return error;
                  }

                  return (
                    error.msg ||
                    JSON.stringify(error)
                  );
                })
                .join(", ");
          } else if (
            uploadResult.detail
          ) {
            errorMessage =
              JSON.stringify(
                uploadResult.detail
              );
          }

          throw new Error(
            errorMessage
          );
        }

        console.log(
          `Uploaded ${file.name}:`,
          uploadResult
        );
      }

      if (question) {
        const response = await fetch(
          `${API_URL}/chat`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              chat_id: currentChatId,
              question,
            }),
          }
        );

        const result =
          await response.json();

        if (!response.ok) {
          let errorMessage =
            "Failed to get response";

          if (
            typeof result.detail === "string"
          ) {
            errorMessage =
              result.detail;
          } else if (result.detail) {
            errorMessage =
              JSON.stringify(
                result.detail
              );
          }

          throw new Error(
            errorMessage
          );
        }

        if (result.blocked) {
          window.alert(
            result.message ||
              "Please enter a valid medical question."
          );

          return;
        }

        localStorage.setItem(
          "chat_id",
          currentChatId
        );

        if (onNewQuery) {
          onNewQuery(
            currentChatId,
            question,
            result
          );
        }

        if (onResponse) {
          onResponse(result);
        }

        window.dispatchEvent(
          new Event(
            "evaluation-created"
          )
        );
      }

      setMessage("");
      setFiles([]);

      if (textareaRef.current) {
        textareaRef.current.style.height =
          "auto";
      }
    } catch (error) {
      console.error(
        "Chat/upload error:",
        error
      );

      window.alert(
        error instanceof Error
          ? error.message
          : "Sorry, something went wrong while contacting the server."
      );

      if (onResponse) {
        onResponse({
          answer:
            "Sorry, something went wrong while contacting the server.",
        });
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (
      e.key === "Enter" &&
      !e.shiftKey
    ) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <>
      {files.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {files.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className="flex max-w-[240px] items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 px-2 py-1.5"
            >
              <span className="shrink-0">
                {file.type ===
                "application/pdf"
                  ? "📄"
                  : "🖼️"}
              </span>

              <span
                className="min-w-0 truncate text-sm text-gray-700"
                title={file.name}
              >
                {file.name}
              </span>

              <button
                type="button"
                onClick={() =>
                  removeFile(index)
                }
                className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-purple-900 text-white"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {loading && (
        <div className="mb-3 flex items-center gap-2 px-2 text-sm text-gray-500">
          <span>
            Generating response
          </span>

          <span className="flex gap-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-500 [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-500 [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-500" />
          </span>
        </div>
      )}

      <div className="flex items-end gap-2">
        <button
          type="button"
          onClick={() =>
            fileInputRef.current?.click()
          }
          disabled={loading}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-gray-500 hover:bg-primary hover:text-white disabled:opacity-50"
        >
          <svg
            className="h-5 w-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
          </svg>
        </button>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.jpg,.jpeg,.png,.webp,application/pdf,image/jpeg,image/png,image/webp"
          className="hidden"
          onChange={handleFiles}
        />

        <textarea
          ref={textareaRef}
          value={message}
          onChange={handleMessageChange}
          onKeyDown={handleKeyDown}
          rows={1}
          style={{
            fontFamily: "sans-serif",
          }}
          placeholder="Ask medical related questions"
          disabled={loading}
          className="max-h-44 min-h-10 flex-1 resize-none overflow-y-auto bg-transparent px-1 py-2.5 text-lg leading-5 outline-none placeholder:text-gray-600"
        />

        <button
          type="button"
          onClick={handleSubmit}
          disabled={
            !canSend || loading
          }
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary text-white hover:bg-gray-800 disabled:bg-gray-200 disabled:text-gray-400"
        >
          {loading ? (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-gray-400 border-t-transparent" />
          ) : (
            "↑"
          )}
        </button>
      </div>
    </>
  );
}