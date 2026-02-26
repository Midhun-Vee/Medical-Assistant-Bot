"use client";

export default function ChatHistory({
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
}) {
  return (
    <aside className="ml-4 mt-8 flex h-full w-72 flex-col rounded-lg border border-primary/40 bg-white/15 shadow-[0_8px_32px_0_rgba(0,0,0,0.37)] backdrop-blur-[50px] [-webkit-backdrop-filter:blur(10px)]">
      <div className="border-b border-white/40 bg-white/10 p-4 backdrop-blur-2xl">
        <h2 className="text-lg font-semibold text-black">
          Chat History
        </h2>

        <p className="mt-1 text-sm text-black/60">
          Your recent questions
        </p>
      </div>

      <div className="p-3">
        <button
          type="button"
          onClick={onNewChat}
          className="w-full rounded-lg border bg-primary border-primary/60 px-4 py-3 text-sm font-medium text-white shadow-lg backdrop-blur-2xl transition-all duration-200 hover:border-primary hover:bg-white/50 hover:shadow-xl hover:text-primary"
        >
          + New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {chats.length === 0 ? (
          <p className="rounded-lg border border-white/30 px-3 py-4 text-sm text-black/50">
            No conversations yet.
          </p>
        ) : (
          <div className="space-y-2">
            {chats.map((chat) => {
              const isActive = chat.id === activeChatId;

              return (
                <button
                  key={chat.id}
                  type="button"
                  onClick={() => onSelectChat(chat.id)}
                  className={`w-full rounded-lg border px-3 py-3 text-left backdrop-blur-2xl transition-all duration-200 ${
                    isActive
                      ? "border-primary bg-white/50 text-black shadow-lg"
                      : "border-primary bg-white/15 text-black/70 hover:border-white/60 hover:bg-white/35 hover:text-black"
                  }`}
                >
                  <p className="line-clamp-2 text-sm">
                    {chat.title}
                  </p>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}