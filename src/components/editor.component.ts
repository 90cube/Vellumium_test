
import { Component, inject } from '@angular/core';
import { CommonModule, NgOptimizedImage } from '@angular/common';
import { VellumiumService, Scene } from '../services/vellumium.service';

@Component({
  selector: 'app-editor',
  standalone: true,
  imports: [CommonModule, NgOptimizedImage],
  template: `
    <div class="min-h-screen bg-zinc-900 flex flex-col items-center pt-8 pb-20 relative">
      
      <!-- Top Controls -->
      <div class="fixed top-0 left-0 right-0 z-40 bg-zinc-950/90 backdrop-blur-sm border-b border-zinc-800 p-4 flex justify-between items-center">
        <button (click)="vellumService.backToDashboard()" class="text-zinc-400 hover:text-white flex items-center gap-2 font-script">
          <span>&larr; Eject Tape</span>
        </button>
        <h1 class="font-cinema text-2xl text-amber-500">{{ vellumService.selectedProject()?.title }}</h1>
        <div class="w-24"></div> <!-- Spacer -->
      </div>

      <!-- Film Strip Container -->
      <div class="mt-20 w-full max-w-2xl relative">
        
        <!-- Film Sprockets Background Pattern -->
        <div class="absolute left-0 top-0 bottom-0 w-12 bg-black border-r border-zinc-800 flex flex-col gap-4 py-4 items-center overflow-hidden">
           @for(i of sprockets; track i) { <div class="w-6 h-4 bg-zinc-800 rounded-sm"></div> }
        </div>
        <div class="absolute right-0 top-0 bottom-0 w-12 bg-black border-l border-zinc-800 flex flex-col gap-4 py-4 items-center overflow-hidden">
           @for(i of sprockets; track i) { <div class="w-6 h-4 bg-zinc-800 rounded-sm"></div> }
        </div>

        <!-- Scenes List -->
        <div class="px-16 py-8 flex flex-col gap-8 bg-zinc-900/50 min-h-screen border-l border-r border-zinc-800">
          
          @for (scene of vellumService.selectedProject()?.scenes; track scene.id) {
            <div 
              class="relative aspect-video bg-black cursor-pointer group border-4 border-black hover:border-amber-500/50 transition-all duration-300 shadow-2xl"
              (click)="openScene(scene)"
            >
              <img [ngSrc]="scene.thumbnail" fill alt="Scene" class="object-cover opacity-80 group-hover:opacity-100 transition-opacity">
              
              <!-- Hover Overlay -->
              <div class="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 flex flex-col items-center justify-center transition-opacity duration-300">
                <span class="font-cinema text-2xl text-amber-500 tracking-widest">EDIT SCENE</span>
                <span class="font-script text-xs text-zinc-400 mt-2">{{ scene.title }}</span>
              </div>
              
              <!-- Scene Number -->
              <div class="absolute -left-12 top-1/2 -translate-y-1/2 text-zinc-600 font-mono text-xl -rotate-90">
                SC-{{scene.id}}
              </div>
            </div>
            
            <!-- Connection Line -->
            <div class="h-8 w-1 bg-zinc-800 mx-auto"></div>
          }

          <!-- Add Scene Button -->
          <div 
            class="relative aspect-video bg-zinc-800/30 border-2 border-dashed border-zinc-700 flex flex-col items-center justify-center cursor-pointer hover:bg-zinc-800/50 hover:border-amber-500/30 transition-all"
            (click)="addScene()"
          >
             <span class="text-4xl text-zinc-600 mb-2">+</span>
             <span class="font-script text-zinc-500">Add Scene Frame</span>
          </div>

        </div>
      </div>
    </div>
  `
})
export class EditorComponent {
  vellumService = inject(VellumiumService);
  // Just for rendering sprocket holes visual
  sprockets = Array(50).fill(0).map((x, i) => i); 

  openScene(scene: Scene) {
    this.vellumService.selectScene(scene);
  }

  addScene() {
    // Logic to add a new scene (in a real app, this would modify the signal)
    alert("New scene strip added (Mock)");
  }
}
