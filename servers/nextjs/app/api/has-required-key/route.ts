import { NextResponse } from "next/server";
import fs from "fs";

export const dynamic = "force-dynamic";

export async function GET() {
  const userConfigPath = process.env.USER_CONFIG_PATH;

  let config = {};
  if (userConfigPath && fs.existsSync(userConfigPath)) {
    try {
      const raw = fs.readFileSync(userConfigPath, "utf-8");
      config = JSON.parse(raw || "{}");
    } catch {}
  }

  // Check for any available API keys
  const openaiKey = config.OPENAI_API_KEY || process.env.OPENAI_API_KEY || "";
  const customKey = config.CUSTOM_LLM_API_KEY || process.env.CUSTOM_LLM_API_KEY || "";
  const googleKey = config.GOOGLE_API_KEY || process.env.GOOGLE_API_KEY || "";
  const anthropicKey = config.ANTHROPIC_API_KEY || process.env.ANTHROPIC_API_KEY || "";
  
  // Check LLM provider preference
  const llmProvider = config.LLM || process.env.LLM || "";
  
  let hasKey = false;
  
  // Check based on provider preference or any available key
  if (llmProvider === "custom" && customKey.trim()) {
    hasKey = true;
  } else if (llmProvider === "openai" && openaiKey.trim()) {
    hasKey = true;
  } else if (llmProvider === "google" && googleKey.trim()) {
    hasKey = true;
  } else if (llmProvider === "anthropic" && anthropicKey.trim()) {
    hasKey = true;
  } else if (openaiKey.trim() || customKey.trim() || googleKey.trim() || anthropicKey.trim()) {
    // Fallback: if any key is available
    hasKey = true;
  }

  console.log(`LLM Provider: ${llmProvider}, Has Key: ${hasKey}`);

  return NextResponse.json({ hasKey });
} 