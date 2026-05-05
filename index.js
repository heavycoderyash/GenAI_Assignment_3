import readline from "readline";
import fs from "fs";
import path from "path";
import { exec } from "child_process";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const client = new OpenAI({
  apiKey: process.env.API_KEY,
  baseURL: process.env.BASE_URL,
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

let messages = [
  {
    role: "system",
    content: `
You are an expert AI coding agent that builds a high-fidelity frontend clone of the Scaler Academy "Modern Software & AI Engineering" homepage step-by-step.

Goal: produce index.html, styles.css, and script.js that visually resemble the Scaler site based on the specific design system and copy provided below.

-----------------------------------
HOW YOU SHOULD WORK
-----------------------------------
- Think step-by-step before taking actions
- Perform ONE action at a time (one file per step)
- Order: index.html → styles.css → script.js → optional polish pass
- Only emit ACTION: done after all three files exist and the page looks complete

-----------------------------------
OUTPUT FORMAT (STRICT)
-----------------------------------
THOUGHT: <what you plan to do>
ACTION: <action_name>
DATA:
<filename>
<file content>

-----------------------------------
CRITICAL FORMATTING RULES
-----------------------------------
- DO NOT use markdown code blocks (no \`\`\`)
- First line after DATA = filename only (e.g. index.html)
- Rest = raw file contents
- No commentary after the file body

-----------------------------------
AVAILABLE ACTIONS
-----------------------------------
- create_file
- done

-----------------------------------
VISUAL REFERENCE & DESIGN SYSTEM
-----------------------------------
- Typography: Use 'Plus Jakarta Sans', 'Clash Grotesk', or system sans-serif (Inter/Segoe UI). 
- Primary Brand Blue: #004CE5 (use for primary CTAs and active states).
- Backgrounds: Mostly white (#FFFFFF) and light grays (#F5F5F5), with specific dark Navy/Black sections where noted.
- Text Colors: Deep dark gray/black (#121212) for primary text on light backgrounds.
- UI Elements: Modern cards with subtle borders (#E4E4E4), 8px-12px border-radius, soft shadows.
- Buttons: 
  - Primary: Solid #004CE5 background, white text.
  - Outline: Transparent background, border #E4E4E4 or #004CE5, dark text.

-----------------------------------
PAGE STRUCTURE & COPY (Must replicate these 5 sections exactly)
-----------------------------------
1. Top Navbar (Sticky, white background, flexbox, thin bottom border)
   - Left: SCALER text logo.
   - Center links: PROGRAMS ▾, MASTERCLASS, AI LABS, ALUMNI, RESOURCES ▾
   - Right: "Login" (Outline Button) + "Request A Callback" (Solid Blue Button)

2. Hero Section (White background, center-aligned flex/grid)
   - Subline: "Software Engineering hasn't changed. What it takes to be a great at it has. Stronger fundamentals, faster delivery, and AI fluency built into how you learn, not just an add on" (Muted text, bold the first sentence).
   - Massive Headline: "Modern Software and AI Engineering." (Make "AI Engineering." blue to stand out).
   - Features Row (Flex row, gap 20px, center aligned): 
     - 3 items: "AI-Integrated Curriculum", "Unlock Lifelong Learning", "Fundamentals First". 
     - ABOVE each text item, place a small circular blue div (width/height ~32px, border-radius 50%, background #e6f0ff, color #004ce5, center content) containing the ✨ emoji.
   - Meta tag below: "Next cohort starts May 2026"
   - Buttons (Flex row, gap 16px, center aligned): "Download Brochure" (Solid Blue, large) + "Talk to an advisor" (Outline, large).

3. "Who it is for?" Section (Light gray background)
   - Headline: "AI is reshaping every role. Are you ready for yours?"
   - Subtitle: "The problem isn't talent - it's that your current stack doesn't include the AI layer companies now expect."
   - Display a grid of 4 Persona Cards (White background, padded, subtle border):
     1. SDEs, QA Engineers & Frontend (1-4 Years)
     2. Fullstack Developers, Backend & API (2-6 Years)
     3. Software Engineers & Tech Leads (3-8 Years)
     4. Solution Engineers & AI Builders (5-10+ Years)

4. "Why Scaler" Section (Dark Navy / Black background, white text)
   - Eyebrow: "AI is the operating system"
   - Headline: "Software Engineering Has Changed. So Has This Program."
   - Subtitle: "Other programmes teach you to ship features. We teach you to build, evaluate, and own AI-integrated systems end to end."
   - Grid of 4 feature cards (Dark gray cards with white text):
     • AI-Led Curriculum
     • AI Mock Interviews
     • AI-Driven Project Portfolio
     • AI Learning Model

5. Footer (Simple light grey footer)
   - SCALER Logo, copyright text, and simple text links.

-----------------------------------
HARD RULES
-----------------------------------
- Do NOT use a dark theme overall — only the "Why Scaler" section is dark.
- Do NOT leave any section unstyled.
- Do NOT emit ACTION: done before index.html, styles.css, AND script.js all exist.
- Do NOT use long text labels like "[Sparkle Icon]" inside small CSS shapes. Use actual Unicode emojis (like ✨) or clean CSS geometry so the layout does not break.
- For large images, use solid color placeholder divs.
- Keep it a realistic single-author build (no external frameworks, no Tailwind, no CDN UI kits).

Start by creating index.html with the full semantic skeleton for all 5 sections above.
`
  }
];

let filesCreated = new Set();

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function openInBrowser(fileName) {
  return new Promise((resolve) => {
    const fullPath = path.resolve(fileName);
    if (!fs.existsSync(fullPath)) {
      console.log(`Cannot open ${fileName} — file not found.`);
      return resolve();
    }

    const cmd =
      process.platform === "darwin"
        ? `open "${fullPath}"`
        : process.platform === "win32"
        ? `start "" "${fullPath}"`
        : `xdg-open "${fullPath}"`;

    exec(cmd, (err) => {
      if (err) console.log(`Failed to open browser: ${err.message}`);
      else console.log(`Opened ${fileName} in browser.`);
      resolve();
    });
  });
}

const TYPE_DELAY_MS = 5;

async function typeWrite(text, delay = TYPE_DELAY_MS) {
  for (const char of text) {
    process.stdout.write(char);
    await sleep(delay);
  }
}

async function runAgent() {
  const stream = await client.chat.completions.create({
    model: process.env.MODEL,
    messages: messages,
    stream: true,
  });

  process.stdout.write("\nLLM output:\n");

  let output = "";
  for await (const chunk of stream) {
    const token = chunk.choices?.[0]?.delta?.content || "";
    if (token) {
      output += token;
      for (const char of token) {
        process.stdout.write(char);
        await sleep(TYPE_DELAY_MS);
      }
    }
  }
  process.stdout.write("\n");

  return output;
}

function parseOutput(output) {
  const thoughtMatch = output.match(/THOUGHT:\s*(.*)/);
  const actionMatch = output.match(/ACTION:\s*(\w+)/);
  const dataMatch = output.match(/DATA:\s*([\s\S]*)/);

  return {
    thought: thoughtMatch ? thoughtMatch[1].trim() : null,
    action: actionMatch ? actionMatch[1] : null,
    data: dataMatch ? dataMatch[1].trim() : null,
  };
}

function createFile(data) {
  data = data.replace(/```[a-z]*\n?/gi, "").replace(/```/g, "");

  const firstLineEnd = data.indexOf("\n");

  if (firstLineEnd === -1) {
    console.log("Invalid DATA format");
    return null;
  }

  let fileName = data.substring(0, firstLineEnd).trim();
  let content = data.substring(firstLineEnd + 1).trim();

  if (fileName.startsWith("html")) fileName = "index.html";
  if (fileName.startsWith("css")) fileName = "styles.css";

  fs.writeFileSync(fileName, content);
  console.log(`File ${fileName} created`);

  filesCreated.add(fileName);
  return fileName;
}

async function agentLoop(userInput) {
  let currentInput = userInput;
  let steps = 0;
  const MAX_STEPS = 10; 

  while (true) {
    steps++;

    if (steps > MAX_STEPS) {
      console.log("Max steps reached. Stopping agent.");
      break;
    }

    messages.push({ role: "user", content: currentInput });

    const output = await runAgent();
    messages.push({ role: "assistant", content: output });

    const { thought, action, data } = parseOutput(output);

    if (action === "create_file") {
      const fileName = createFile(data);

      currentInput = `File ${fileName} created. Continue.`;

      if (
        filesCreated.has("index.html") &&
        filesCreated.has("styles.css") &&
        filesCreated.has("script.js")
      ) {
        console.log("Required files created. Finishing.");
        await openInBrowser("index.html");
        break;
      }
    }
    else if (action === "done") {
      console.log("Agent is done.");
      if (filesCreated.has("index.html")) await openInBrowser("index.html");
      break;
    }
    else {
      console.log("Unknown action.");
      break;
    }
  }
}

rl.question(">> ", async (input) => {
  await agentLoop(input);
  rl.close();
  process.exit(0);
});