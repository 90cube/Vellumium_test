
import { Injectable, signal, computed } from '@angular/core';
import { GoogleGenAI } from "@google/genai";

export type ViewState = 'LANDING' | 'DASHBOARD' | 'EDITOR' | 'CANVAS' | 'PREMIERE';
export type ProjectType = 'Film' | 'DirectorCut' | 'Storyboard';

export interface Layer {
  id: string;
  type: 'image' | 'draw' | 'text';
  content: string; 
  zIndex: number;
  locked?: boolean; // For 'Film' type where layers are merged/locked
}

export interface Scene {
  id: string;
  title: string;
  thumbnail: string;
  layers: Layer[];
  description: string;
  generatedVideoUrl?: string; // For DirectorCut/Film
}

export interface Project {
  id: string;
  title: string;
  description: string;
  createdDate: Date;
  scenes: Scene[];
  type: ProjectType;
  genre: string;
}

export interface Post {
  id: string;
  projectId: string;
  projectType: ProjectType;
  author: string;
  title: string;
  videoUrl?: string; 
  thumbnail?: string;
  subtitles?: { time: number; text: string }[];
  likes: number;
  comments: { user: string; text: string }[];
  genre: string;
}

@Injectable({
  providedIn: 'root'
})
export class VellumiumService {
  // State Signals
  readonly currentView = signal<ViewState>('LANDING');
  readonly isLoggedIn = signal<boolean>(false);
  readonly user = signal<{ name: string; avatar: string } | null>(null);
  readonly selectedProject = signal<Project | null>(null);
  readonly selectedScene = signal<Scene | null>(null);

  // Mock Data
  readonly projects = signal<Project[]>([
    {
      id: 'p1',
      title: 'Neon Nights',
      description: 'A cyberpunk detective story set in 2084.',
      createdDate: new Date(),
      type: 'Film',
      genre: 'Sci-Fi',
      scenes: [
        { id: 's1', title: 'The Alley', thumbnail: 'https://picsum.photos/seed/neon1/600/400', description: 'Rainy neon alleyway', layers: [] },
        { id: 's2', title: 'The Chase', thumbnail: 'https://picsum.photos/seed/neon2/600/400', description: 'Hovercar chase', layers: [] },
      ]
    },
    {
      id: 'p2',
      title: 'Whispers',
      description: 'A silent film about a ghost.',
      createdDate: new Date(),
      type: 'DirectorCut',
      genre: 'Horror',
      scenes: [
         { id: 's4', title: 'The Door', thumbnail: 'https://picsum.photos/seed/lib1/600/400', description: 'Grand dusty doors', layers: [], generatedVideoUrl: 'https://media.istockphoto.com/id/1363640237/video/dust-particles-in-a-ray-of-light.mp4?s=mp4-640x640-is&k=20&c=0sLgqfO-T7O_0oK6Uo_QhT2FqZl4KyC_4qF_1_2_3_4=' }
      ]
    },
    {
        id: 'p3',
        title: 'Dreamscape',
        description: 'Surreal imagery generated daily.',
        createdDate: new Date(),
        type: 'Storyboard',
        genre: 'Abstract',
        scenes: [
            { id: 's5', title: 'Cloud City', thumbnail: 'https://picsum.photos/seed/cloud/600/400', description: 'Floating islands', layers: [] }
        ]
    }
  ]);

  readonly communityPosts = signal<Post[]>([
    {
      id: 'post1',
      projectId: 'p1',
      projectType: 'Film',
      author: 'CyberDirector',
      title: 'Neon Nights - Final Cut',
      genre: 'Sci-Fi',
      likes: 124,
      videoUrl: 'https://media.istockphoto.com/id/1404169973/video/abstract-digital-blue-color-wave-with-flowing-small-particles-dance-motion-on-wave-and.mp4?s=mp4-640x640-is&k=20&c=M_b6_gOny-5C-E0FfF0_uM3KsmZjJ6T9H-EszFwbJ9Q=', 
      subtitles: [
        { time: 0, text: "[Rain falling heavy]" },
        { time: 2, text: "I knew he would come back..." },
        { time: 5, text: "But not like this." }
      ],
      comments: [
        { user: 'MovieBuff', text: 'The color grading is insane! 🎬' },
        { user: 'Critic101', text: 'Pacing is a bit slow.' }
      ]
    },
    {
        id: 'post2',
        projectId: 'p3',
        projectType: 'Storyboard',
        author: 'ArtsyOne',
        title: 'Dreamscape Concept',
        genre: 'Abstract',
        likes: 45,
        thumbnail: 'https://picsum.photos/seed/cloud/600/400',
        comments: [
            { user: 'Fan1', text: 'Love the brushwork.'}
        ]
    }
  ]);

  constructor() {}

  // Actions
  login() {
    this.isLoggedIn.set(true);
    this.user.set({ name: 'VellumUser', avatar: 'https://picsum.photos/seed/user/50/50' });
  }

  logout() {
    this.isLoggedIn.set(false);
    this.user.set(null);
    this.currentView.set('LANDING');
  }

  navigateTo(view: ViewState) {
    this.currentView.set(view);
  }

  selectProject(project: Project) {
    this.selectedProject.set(project);
    this.currentView.set('EDITOR');
  }

  selectScene(scene: Scene) {
    this.selectedScene.set(scene);
    this.currentView.set('CANVAS');
  }

  backToEditor() {
    this.selectedScene.set(null);
    this.currentView.set('EDITOR');
  }

  backToDashboard() {
    this.selectedProject.set(null);
    this.currentView.set('DASHBOARD');
  }

  // Gemini Integration
  async generateSceneIdea(prompt: string): Promise<string> {
    try {
      const apiKey = process.env['API_KEY'];
      if (!apiKey) return "API Key missing. Simulating AI thought process...";

      const ai = new GoogleGenAI({ apiKey: apiKey });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: `Generate a short, cinematic description for a movie scene based on: ${prompt}. Keep it under 20 words.`
      });
      return response.text.trim();
    } catch (e) {
      console.error(e);
      return "AI generation failed. Try again.";
    }
  }

  async generateImageForScene(prompt: string): Promise<string> {
       try {
        const apiKey = process.env['API_KEY'];
        if (!apiKey) return `https://picsum.photos/seed/${Math.random()}/800/600`;

        const ai = new GoogleGenAI({ apiKey: apiKey });
        const response = await ai.models.generateImages({
            model: 'imagen-4.0-generate-001',
            prompt: prompt,
            config: {
                numberOfImages: 1,
                aspectRatio: '16:9', 
                outputMimeType: 'image/jpeg'
            }
        });
        
        const base64 = response.generatedImages?.[0]?.image?.imageBytes;
        if(base64) {
            return `data:image/jpeg;base64,${base64}`;
        }
        return `https://picsum.photos/seed/${Math.random()}/800/600`;

      } catch (e) {
        console.error("Image gen error", e);
        return `https://picsum.photos/seed/${Math.random()}/800/600`;
      }
  }
}
