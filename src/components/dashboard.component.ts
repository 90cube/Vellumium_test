
import { Component, inject, signal } from '@angular/core';
import { CommonModule, NgOptimizedImage } from '@angular/common';
import { VellumiumService, Project } from '../services/vellumium.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, NgOptimizedImage],
  template: `
    <div class="min-h-screen bg-[#1a1a1a] p-8 pt-20 overflow-x-hidden relative">
      
      <!-- Header -->
      <header class="flex justify-between items-end mb-16 border-b border-zinc-800 pb-4 max-w-6xl mx-auto">
        <div>
           <h2 class="font-cinema text-4xl text-amber-500">The Vault</h2>
           <p class="font-script text-zinc-400 mt-2">Select a tape from the shelf.</p>
        </div>
        <div class="flex gap-4">
           <button (click)="vellumService.navigateTo('PREMIERE')" class="group flex items-center gap-2 px-4 py-2 text-zinc-400 hover:text-amber-500 transition-colors">
             <span class="font-cinema">GO TO THEATER</span>
             <span class="group-hover:translate-x-1 transition-transform">→</span>
           </button>
           <button class="px-6 py-2 bg-amber-900/20 text-amber-500 border border-amber-800 hover:bg-amber-800 hover:text-white font-script text-sm rounded-sm transition-all shadow-[0_0_10px_rgba(245,158,11,0.1)]">
             + REC New Tape
           </button>
        </div>
      </header>

      <!-- VHS Shelf Container -->
      <div class="max-w-6xl mx-auto h-[500px] flex items-end justify-center perspective-1500 gap-1 pb-12 border-b-[20px] border-zinc-800 bg-[url('https://www.transparenttextures.com/patterns/wood-pattern.png')] shadow-[inset_0_-20px_50px_rgba(0,0,0,0.8)] px-12 z-0">
        
        @for (project of vellumService.projects(); track project.id) {
          <!-- The Tape Container -->
          <div 
            class="relative group w-[50px] h-[300px] transition-all duration-500 ease-out preserve-3d cursor-pointer hover:mx-12"
            (click)="expandTape(project)"
          >
             
             <!-- Spine (Visible Initially) -->
             <div class="absolute inset-0 bg-black rounded-sm border-l border-t border-zinc-700 shadow-xl flex flex-col items-center justify-between py-4 z-20 transition-transform duration-500 group-hover:translate-z-20 group-hover:translate-y-[-50px] bg-[url('https://www.transparenttextures.com/patterns/black-linen.png')]">
                
                <!-- Top Sticker -->
                <div class="w-8 h-8 bg-zinc-800 rounded-sm border border-zinc-600"></div>
                
                <!-- Title Label -->
                <div class="h-40 w-10 bg-white shadow-sm flex items-center justify-center transform overflow-hidden rounded-[2px]">
                   <span class="font-script text-black font-bold whitespace-nowrap rotate-90 text-lg tracking-tight truncate w-32 text-center">{{ project.title }}</span>
                </div>

                <!-- Genre/Type Sticker -->
                <div class="w-10 h-10 rounded-full border-2 border-amber-700 flex items-center justify-center bg-amber-900/50">
                   <span class="text-[10px] text-amber-500 font-mono -rotate-90">{{ project.genre.substring(0,3) }}</span>
                </div>

             </div>
             
             <!-- Side Depth (Mock 3D) -->
             <div class="absolute top-0 right-0 w-8 h-full bg-zinc-800 origin-right transform rotate-y-90 translate-x-full border-l border-zinc-900"></div>

          </div>
        }
        
        <!-- Empty Slots Filler -->
        @for(i of [1,2,3,4,5]; track i) {
           <div class="w-[50px] h-[300px] border-l border-r border-zinc-800/30 bg-black/20 transform skew-x-6 opacity-30"></div>
        }

      </div>

      <!-- Expanded Tape Modal (Moved outside to avoid transform clipping) -->
      @if (expandedProject(); as project) {
        <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm" (click)="closeExpansion()">
           
           <!-- The Expanded Tape (Face View) -->
           <div class="relative w-[800px] h-[450px] bg-black border-4 border-zinc-800 rounded-lg shadow-2xl flex overflow-hidden animate-pop-in" (click)="$event.stopPropagation()">
              
              <!-- Left: Tape Mechanics Visual -->
              <div class="w-1/3 bg-zinc-900 relative border-r border-zinc-800 flex flex-col items-center justify-center gap-8 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]">
                 <div class="w-32 h-32 rounded-full border-4 border-zinc-700 bg-white/10 flex items-center justify-center animate-spin-slow">
                    <div class="w-4 h-4 bg-zinc-500 rounded-full"></div>
                    <div class="absolute w-full h-2 bg-transparent border-t border-zinc-600 rotate-45"></div>
                    <div class="absolute w-full h-2 bg-transparent border-t border-zinc-600 -rotate-45"></div>
                 </div>
                 <div class="w-32 h-32 rounded-full border-4 border-zinc-700 bg-white/10 flex items-center justify-center animate-spin-slow">
                    <div class="w-4 h-4 bg-zinc-500 rounded-full"></div>
                 </div>
                 
                 <!-- Info -->
                 <div class="absolute bottom-6 left-6 right-6 text-center">
                    <h3 class="font-cinema text-amber-500 text-2xl mb-2">{{ project.title }}</h3>
                    <p class="font-mono text-zinc-500 text-xs uppercase tracking-widest">{{ project.type }} • {{ project.genre }}</p>
                 </div>
              </div>

              <!-- Right: Scene Preview -->
              <div class="w-2/3 relative bg-black group-hover:bg-zinc-900 transition-colors">
                 <!-- First Scene Image -->
                 @if(project.scenes.length > 0) {
                   <img [ngSrc]="project.scenes[0].thumbnail" fill alt="First Scene" class="object-cover opacity-60">
                 } @else {
                   <div class="w-full h-full flex items-center justify-center text-zinc-700 font-cinema">NO FOOTAGE</div>
                 }
                 
                 <!-- Actions Overlay -->
                 <div class="absolute inset-0 bg-gradient-to-t from-black via-black/50 to-transparent flex flex-col justify-end p-8">
                    <h4 class="font-cinema text-white text-xl mb-2">Synopsis</h4>
                    <p class="font-script text-zinc-300 text-sm mb-8 line-clamp-3 leading-relaxed">{{ project.description }}</p>
                    
                    <div class="flex gap-4">
                       <button (click)="enterEditor(project)" class="flex-1 py-4 bg-amber-600 hover:bg-amber-500 text-black font-cinema font-bold text-xl rounded-sm transition-colors tracking-widest flex items-center justify-center gap-2">
                         <span>▶</span> LOAD TAPE
                       </button>
                       @if(project.type === 'Film') {
                         <button class="px-6 py-4 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-mono text-xs rounded-sm border border-zinc-600 uppercase tracking-widest">
                           Auto-Gen
                         </button>
                       }
                    </div>
                 </div>
              </div>

              <!-- Close Button -->
              <button (click)="closeExpansion()" class="absolute top-4 right-4 text-zinc-500 hover:text-white z-10 p-2">
                ✕
              </button>

           </div>
        </div>
      }

    </div>
  `,
  styles: [`
    .perspective-1500 {
      perspective: 1500px;
    }
    .preserve-3d {
      transform-style: preserve-3d;
    }
    .translate-z-20 {
      transform: translateZ(20px);
    }
    .animate-pop-in {
      animation: popIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }
    .animate-spin-slow {
      animation: spin 10s linear infinite;
    }
    @keyframes popIn {
      from { opacity: 0; transform: scale(0.9) translateY(20px); }
      to { opacity: 1; transform: scale(1) translateY(0); }
    }
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
  `]
})
export class DashboardComponent {
  vellumService = inject(VellumiumService);
  expandedProject = signal<Project | null>(null);

  expandTape(project: Project) {
    this.expandedProject.set(project);
  }

  closeExpansion() {
    this.expandedProject.set(null);
  }

  enterEditor(project: Project) {
    this.vellumService.selectProject(project);
  }
}
