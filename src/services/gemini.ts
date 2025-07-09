import { GoogleGenAI } from '@google/genai'
import { env } from '../env.ts'

const gemini = new GoogleGenAI({
  apiKey: env.GEMINI_API_KEY,
})

const model = 'gemini-2.5-flash'

export async function transcribeAudio(audioAsBase64: string, mimeType: string) {
  const response = await gemini.models.generateContent({
    model,
    contents: [
      {
        text: 'Transcribe the following audio. Be precise and natural in the transcription. Maintain proper ponctuation and split the text into paragraphs when needed.',
      },
      {
        inlineData: {
          mimeType,
          data: audioAsBase64,
        },
      },
    ],
  })

  if (!response.text) {
    throw new Error('Failed to transcribe audio')
  }

  return response.text
}

export async function generateEmbeddings(text: string) {
  const response = await gemini.models.embedContent({
    model: 'text-embedding-004',
    contents: [
      {
        text,
      },
    ],
    config: {
      taskType: 'RETRIEVAL_DOCUMENT',
    },
  })

  if (!response.embeddings?.[0].values) {
    throw new Error('Failed to generate embeddings')
  }

  return response.embeddings[0].values
}

export async function generateAnswer(
  question: string,
  transcriptions: string[]
) {
  const context = transcriptions.join('\n\n')

  const prompt = `
    Based on the text below, answer the question in a clear and precise way in the same language the question was made.

    CONTEXT:
    ${context}

    QUESTION:
    ${question}

    INSTRUCTIONS:
    - Use only informations provided by the context above.
    - IF the answer cant be found in the context just answer that you couldn't find enough information to answer.
    - Be objective.
    - Maintain a professional and polite tone.
    - Quote the context when necessary.
    - If you are quoting the context, use the term "Class content:" ou "Conteúdo da aula:" before the quote.
    - You must translate all of the answer content into the questions language - it may be English or Brazilian Portuguese
  `.trim()

  const response = await gemini.models.generateContent({
    model,
    contents: [
      {
        text: prompt,
      },
    ],
  })

  if (!response.text) {
    throw new Error('Failed to generate answer')
  }

  return response.text
}
