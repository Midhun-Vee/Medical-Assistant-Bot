"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

import { CardSpotlight } from "@/components/ui/card-spotlight";
import ChatInput from "@/components/ui/Chatinput";
import ChatHistory from "@/components/ui/Chathistory";

import pulse_icon from "@/public/assets/pulse_icon.webp";
import docs_icon from "@/public/assets/docs_icon.webp";

export default function Home() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const storedChats = localStorage.getItem("mediassist_chats");
    const storedActiveChatId = localStorage.getItem("chat_id");

    if (storedChats) {
      try {
        const parsedChats = JSON.parse(storedChats);

        setChats(parsedChats);

        if (storedActiveChatId) {
          const chatExists = parsedChats.some(
            (chat) => chat.id === storedActiveChatId
          );

          if (chatExists) {
            setActiveChatId(storedActiveChatId);
          } else if (parsedChats.length > 0) {
            setActiveChatId(parsedChats[0].id);
          }
        } else if (parsedChats.length > 0) {
          setActiveChatId(parsedChats[0].id);
        }
      } catch (error) {
        console.error("Failed to load chat history:", error);
      }
    }

    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!loaded) {
      return;
    }

    localStorage.setItem(
      "mediassist_chats",
      JSON.stringify(chats)
    );
  }, [chats, loaded]);

  const handleNewChat = () => {
    const chatId = crypto.randomUUID();

    const newChat = {
      id: chatId,
      title: "New Chat",
      messages: [],
    };

    setChats((current) => [...current, newChat]);
    setActiveChatId(chatId);

    localStorage.setItem("chat_id", chatId);
  };

  const handleSelectChat = (chatId) => {
    setActiveChatId(chatId);
    localStorage.setItem("chat_id", chatId);
  };

  const handleNewQuery = (chatId, question, result) => {
    let finalChatId = chatId;

    if (!finalChatId) {
      finalChatId = crypto.randomUUID();
    }

    const userMessage = {
      role: "user",
      content: question,
    };

    const assistantMessage = {
      role: "assistant",
      content: result.answer,
      retrieved_chunks: result.retrieved_chunks,
      judge_verdict: result.judge_verdict,
      judge_block_reason: result.judge_block_reason,
    };

    setChats((current) => {
      const existingChat = current.find(
        (chat) => chat.id === finalChatId
      );

      if (!existingChat) {
        return [
          ...current,
          {
            id: finalChatId,
            title: question,
            messages: [
              userMessage,
              assistantMessage,
            ],
          },
        ];
      }

      return current.map((chat) => {
        if (chat.id !== finalChatId) {
          return chat;
        }

        return {
          ...chat,
          title:
            chat.messages.length === 0
              ? question
              : chat.title,
          messages: [
            ...chat.messages,
            userMessage,
            assistantMessage,
          ],
        };
      });
    });

    setActiveChatId(finalChatId);

    localStorage.setItem(
      "chat_id",
      finalChatId
    );

    if (
      result.evaluation &&
      result.evaluation.status === "evaluated"
    ) {
      localStorage.setItem(
        "has_evaluation",
        "true"
      );

      window.dispatchEvent(
        new Event("evaluation-created")
      );
    }
  };

  if (!loaded) {
    return null;
  }

  const activeChat = chats.find(
    (chat) => chat.id === activeChatId
  );

  const hasStartedChat = chats.length > 0;

  if (!hasStartedChat) {
    return (
      <section className="flex min-h-full flex-1 flex-col">
        <div className="flex flex-1 flex-col items-center justify-center">
          <h1 className="mt-12 text-center text-4xl font-bold text-primary">
            Welcome to MediAssist Clinical AI
          </h1>

          <p className="mt-4 max-w-2xl text-center text-md font-light text-gray-600">
            Your professional clinical decision support tool.
            I can assist with analyzing symptoms, interpreting
            medical reports, and providing evidence-based
            reference information.
          </p>

          <div className="mt-8 flex flex-row gap-10">
            <CardSpotlight className="h-32 w-96 border-gray-400 bg-white">
              <Image
                src={pulse_icon}
                alt="Analyze symptoms"
                className="h-8 w-8"
              />

              <p className="relative z-20 mt-2 text-xl font-bold text-black">
                Analyze Symptoms
              </p>

              <p className="relative z-20 mt-2 text-sm text-black">
                "What could these symptoms indicate?"
              </p>
            </CardSpotlight>

            <CardSpotlight className="h-32 w-96 border-gray-400 bg-white">
              <Image
                src={docs_icon}
                alt="Interpret reports"
                className="h-8 w-8"
              />

              <p className="relative z-20 mt-2 text-xl font-bold text-black">
                Interpret Reports
              </p>

              <p className="relative z-20 mt-2 text-sm text-black">
                "Can you explain this medical report?"
              </p>
            </CardSpotlight>
          </div>
        </div>

        <div className="w-full px-6 pb-6">
          <div className="mx-auto w-full max-w-4xl rounded-lg border border-primary p-2">
            <ChatInput
              chatId={null}
              onNewQuery={handleNewQuery}
            />
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="flex min-h-full flex-1">
      <ChatHistory
        chats={chats}
        activeChatId={activeChatId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-4xl px-6 py-10">
            {activeChat?.messages.map(
              (message, index) => {
                if (message.role === "user") {
                  return (
                    <div
                      key={index}
                      className="mb-8 flex justify-end"
                    >
                      <div className="max-w-2xl rounded-2xl rounded-br-md bg-primary px-5 py-3 text-white">
                        <p className="whitespace-pre-wrap">
                          {message.content}
                        </p>
                      </div>
                    </div>
                  );
                }

                return (
                  <div
                    key={index}
                    className="mb-8 flex justify-start"
                  >
                    <div className="max-w-3xl rounded-2xl rounded-bl-md bg-gray-800 px-5 py-4 text-gray-100">
                      {message.retrieved_chunks !==
                        undefined && (
                        <p className="mb-3 text-xs text-gray-500">
                          Retrieved chunks:{" "}
                          {message.retrieved_chunks}
                        </p>
                      )}

                      <p className="whitespace-pre-wrap">
                        {message.content}
                      </p>

                      {message.judge_verdict && (
                        <div className="mt-5 border-t border-gray-700 pt-4">
                          <p className="font-semibold">
                            Safety Verdict
                          </p>

                          <p className="mt-1 text-sm text-gray-300">
                            {message.judge_verdict}
                          </p>

                          {message.judge_block_reason && (
                            <p className="mt-2 text-sm text-red-400">
                              {message.judge_block_reason}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              }
            )}
          </div>
        </div>

        <div className="mx-auto w-full max-w-4xl px-6 pb-6">
          <div className="w-full rounded-lg border border-primary p-2">
            <ChatInput
              chatId={activeChatId}
              onNewQuery={handleNewQuery}
            />
          </div>
        </div>
      </main>
    </section>
  );
}