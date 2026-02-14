
import { Component, inject, signal } from '@angular/core';
import { CommonModule, NgOptimizedImage } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VellumiumService } from '../services/vellumium.service';

@Component({
  selector: 'app-canvas',
  standalone: true,
  imports: [CommonModule, NgOptimizedImage, FormsModule],
  template: `
    <div class="h-screen flex flex-col bg-zinc-900 overflow-hidden">
      
      <!-- Toolbar -->
      <div class="h-16 bg-zinc-950 border-b border-zinc-800 flex justify-between items-center px-6 shrink-0">
        <div class="flex items-center gap-4">
          <button (click)="vellumService.backToEditor()" class="text-zinc-400 hover:text-white font-script text-sm">
            &larr; Back to Strip
          </button>
          <div class="h-6 w-px bg-zinc-800"></div>
          <h2 class="font-cinema text-amber-500">{{ vellumService.selectedScene()?.title }}</h2>
        </div>
        
        <div class="flex gap-2">
           <button class="px-3 py-1 bg-zinc-800 text-xs font-mono text-zinc-300 rounded hover:bg-zinc-700">Save</button>
           <button class="px-3 py-1 bg-amber-700 text-xs font-mono text-white rounded hover:bg-amber-600">Export</button>
        </div>
      </div>

      <div class="flex-1 flex overflow-hidden">
        
        <!-- Tools Palette -->
        <div class="w-16 bg-zinc-950 border-r border-zinc-800 flex flex-col items-center py-4 gap-4">
           <button class="w-10 h-10 rounded bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700" title="Brush">🖌️</button>
           <button class="w-10 h-10 rounded bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700" title="Text">T</button>
           <button class="w-10 h-10 rounded bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700" title="Shape">⬜</button>
        </div>

        <!-- Main Canvas Area -->
        <div class="flex-1 bg-zinc-900 relative flex items-center justify-center p-8 bg-[radial-gradient(#27272a_1px,transparent_1px)] [background-size:16px_16px]">
          
          <div class="relative shadow-2xl bg-black aspect-video w-full max-w-4xl border border-zinc-700 overflow-hidden group">
             <!-- Base Image -->
             @if (vellumService.selectedScene()?.thumbnail) {
               <img [src]="vellumService.selectedScene()?.thumbnail" class="w-full h-full object-cover">
             }
             
             <!-- Layers (Mock rendering) -->
             @for (layer of layers(); track layer.id) {
               <div class="absolute inset-0 pointer-events-none border border-blue-500/50 opacity-50"></div>
             }

             <!-- Loading Overlay -->
             @if (isGenerating()) {
                <div class="absolute inset-0 bg-black/80 flex flex-col items-center justify-center z-50">
                  <div class="w-12 h-12 border-4 border-amber-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                  <span class="font-script text-amber-500 animate-pulse">Generating Vision...</span>
                </div>
             }
          </div>

        </div>

        <!-- Right Panel: Layers & GenAI -->
        <div class="w-80 bg-zinc-950 border-l border-zinc-800 flex flex-col">
          
          <!-- GenAI Section -->
          <div class="p-4 border-b border-zinc-800">
            <h3 class="font-cinema text-amber-500 mb-2">Director's AI</h3>
            <div class="flex flex-col gap-2">
              <textarea 
                [(ngModel)]="prompt" 
                class="w-full h-24 bg-zinc-900 border border-zinc-700 rounded p-2 text-xs text-white font-mono focus:border-amber-500 focus:outline-none resize-none"
                placeholder="Describe the scene change..."></textarea>
              <button 
                (click)="generateContent()"
                [disabled]="isGenerating()"
                class="w-full py-2 bg-gradient-to-r from-amber-700 to-amber-600 text-white font-cinema text-sm rounded shadow hover:from-amber-600 hover:to-amber-500 disabled:opacity-50">
                {{ isGenerating() ? 'Thinking...' : 'Generate Layer' }}
              </button>
            </div>
          </div>

          <!-- Layers Panel -->
          <div class="flex-1 p-4 overflow-y-auto">
            <h3 class="font-script text-xs text-zinc-500 uppercase tracking-widest mb-4">Layers</h3>
            <div class="space-y-2">
              <div class="p-2 bg-zinc-800/50 border border-zinc-700 rounded flex justify-between items-center text-xs text-zinc-300">
                <span>Base Image</span>
                <span>🔒</span>
              </div>
              <div class="p-2 bg-zinc-900 border border-zinc-800 rounded flex justify-between items-center text-xs text-zinc-500 hover:bg-zinc-800 cursor-pointer">
                <span>+ New Layer</span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  `
})
export class CanvasComponent {
  vellumService = inject(VellumiumService);
  prompt = '';
  isGenerating = signal(false);
  layers = signal<{id: string}[]>([]);

  async generateContent() {
    if (!this.prompt.trim()) return;
    
    this.isGenerating.set(true);
    
    // Simulate generation delay or use Service
    try {
        const result = await this.vellumService.generateSceneIdea(this.prompt);
        this.prompt = result; // Put the refined description back
        
        // Mock creating a layer
        setTimeout(() => {
           this.isGenerating.set(false);
           // In a real app, we'd add the generated image here
           this.layers.update(prev => [...prev, { id: Date.now().toString() }]);
        }, 1500);

    } catch (e) {
        this.isGenerating.set(false);
    }
  }
}
