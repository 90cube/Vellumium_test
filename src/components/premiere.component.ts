
import { Component, inject, signal } from '@angular/core';
import { CommonModule, NgOptimizedImage } from '@angular/common';
import { VellumiumService, Post } from '../services/vellumium.service';

@Component({
  selector: 'app-premiere',
  standalone: true,
  imports: [CommonModule, NgOptimizedImage],
  template: `
    <div class="min-h-screen bg-black text-white pb-20 font-sans">
      
      <!-- Theater Nav -->
      <nav class="sticky top-0 z-40 bg-black/95 backdrop-blur border-b border-amber-900/30 p-4 flex justify-between items-center shadow-lg">
        <div class="flex items-center gap-6">
           <h1 class="font-cinema text-amber-500 text-3xl tracking-widest drop-shadow-[0_0_10px_rgba(245,158,11,0.5)]">PREMIERE</h1>
           <div class="h-6 w-px bg-zinc-800"></div>
           <div class="flex gap-4 text-xs font-script text-zinc-400">
             <button class="hover:text-amber-400 transition-colors uppercase tracking-widest">Featured</button>
             <button class="hover:text-amber-400 transition-colors uppercase tracking-widest">Noir</button>
             <button class="hover:text-amber-400 transition-colors uppercase tracking-widest">Sci-Fi</button>
             <button class="hover:text-amber-400 transition-colors uppercase tracking-widest">Experimental</button>
           </div>
        </div>
        <div class="flex items-center gap-4">
            @if(vellumService.isLoggedIn()) {
                <div class="flex items-center gap-2">
                    <img [src]="vellumService.user()?.avatar" class="w-6 h-6 rounded-full border border-amber-500">
                    <span class="font-script text-xs text-amber-500">{{ vellumService.user()?.name }}</span>
                </div>
            }
            <button (click)="vellumService.backToDashboard()" class="text-xs font-mono text-zinc-500 hover:text-white border border-zinc-800 px-3 py-1 rounded hover:bg-zinc-900 transition-colors">EXIT</button>
        </div>
      </nav>

      <!-- Feed -->
      <div class="max-w-4xl mx-auto mt-12 space-y-32 px-4">
        
        @for (post of vellumService.communityPosts(); track post.id) {
          
          <!-- Only show Storyboard/DirectorCut to logged in users -->
          @if (vellumService.isLoggedIn() || post.projectType === 'Film') {
            
            <article class="flex flex-col gap-6 animate-fade-in">
                
                <!-- Header -->
                <div class="flex justify-between items-end border-b border-zinc-900 pb-2">
                    <div>
                        <h2 class="font-cinema text-4xl text-white">{{ post.title }}</h2>
                        <div class="flex gap-2 mt-1">
                            <span class="px-2 py-0.5 bg-amber-900/30 text-amber-500 text-[10px] font-mono border border-amber-900/50 rounded">{{ post.genre }}</span>
                            <span class="font-script text-zinc-500 text-xs">Dir. {{ post.author }}</span>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        <button class="p-2 text-zinc-500 hover:text-white transition-colors" title="Share SNS">🔗</button>
                    </div>
                </div>

                <!-- Content Viewer -->
                <div class="w-full bg-black shadow-2xl relative group ring-1 ring-zinc-800 rounded-sm overflow-hidden">
                    
                    @if (post.projectType === 'Film') {
                        <!-- Film Viewer (Normal Feed Mode) -->
                        <div 
                          class="aspect-video relative bg-black flex flex-col justify-center cursor-pointer group/video"
                          (click)="openCinemaMode(post)"
                        >
                            <!-- Video Area -->
                            <div class="relative flex-1 bg-zinc-900 overflow-hidden">
                                @if(post.videoUrl) {
                                    <video 
                                        [src]="post.videoUrl" 
                                        loop muted autoplay playsinline 
                                        class="w-full h-full object-cover">
                                    </video>
                                }
                            </div>
                            
                            <!-- Subtitles Overlay (Feed Mode) -->
                            <div class="absolute bottom-8 left-0 right-0 text-center px-8 pointer-events-none">
                                <span class="bg-black/60 text-amber-100/90 font-serif italic text-sm md:text-lg px-2 py-1">
                                    {{ post.subtitles?.[1]?.text || "..." }}
                                </span>
                            </div>

                            <!-- Click Prompt -->
                            <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover/video:opacity-100 transition-opacity bg-black/20">
                                <span class="font-cinema text-white text-xl tracking-widest drop-shadow-md">ENTER CINEMA MODE</span>
                            </div>
                        </div>

                    } @else if (post.projectType === 'DirectorCut') {
                        <!-- Director Cut Viewer (Scrollable Strip) -->
                         <div class="h-96 bg-zinc-900 overflow-y-auto no-scrollbar relative flex flex-col items-center py-8 gap-4 border-l-8 border-r-8 border-black">
                            <div class="sticky top-0 z-20 bg-black/80 px-4 py-1 rounded text-xs text-amber-500 font-mono mb-4">SCROLL TO PREVIEW SCENES</div>
                            <div class="w-3/4 aspect-video bg-zinc-800 border-2 border-zinc-700 hover:scale-105 transition-transform duration-300 flex items-center justify-center text-zinc-600 font-cinema">Scene 1</div>
                            <div class="w-3/4 aspect-video bg-zinc-800 border-2 border-zinc-700 hover:scale-105 transition-transform duration-300 flex items-center justify-center text-zinc-600 font-cinema">Scene 2</div>
                            <div class="w-3/4 aspect-video bg-zinc-800 border-2 border-zinc-700 hover:scale-105 transition-transform duration-300 flex items-center justify-center text-zinc-600 font-cinema">Scene 3</div>
                         </div>

                    } @else if (post.projectType === 'Storyboard') {
                         <!-- Storyboard Viewer (Image Zoom) -->
                         <div class="aspect-video bg-zinc-900 relative overflow-hidden group/sb">
                            <img [src]="post.thumbnail" class="w-full h-full object-contain transition-transform duration-700 group-hover/sb:scale-150 origin-center">
                            <div class="absolute bottom-4 left-4 bg-black/70 p-2 border-l-2 border-amber-500 max-w-md opacity-0 group-hover/sb:opacity-100 transition-opacity delay-200">
                                <p class="font-script text-xs text-white">Concept Art: Initial visualization of the cloud city sequence.</p>
                            </div>
                         </div>
                    }

                </div>

                <!-- Footer / Social -->
                @if (vellumService.isLoggedIn()) {
                    <div class="flex flex-col gap-4 pl-4 border-l border-zinc-800">
                        <div class="flex gap-6 items-center">
                            <button class="flex items-center gap-2 text-zinc-400 hover:text-red-500 transition-colors group">
                                <span class="group-hover:scale-125 transition-transform">♥</span> 
                                <span class="font-mono text-xs">{{ post.likes }} Tickets</span>
                            </button>
                            <button class="flex items-center gap-2 text-zinc-400 hover:text-blue-400 transition-colors">
                                <span>💬</span> 
                                <span class="font-mono text-xs">Critiques</span>
                            </button>
                            <div class="flex-1"></div>
                            <div class="flex gap-2 text-xl grayscale hover:grayscale-0 transition-all cursor-pointer">
                                <span>👏</span><span>🌹</span><span>🍅</span>
                            </div>
                        </div>

                        <!-- Comment Input -->
                        <div class="relative">
                            <input type="text" placeholder="Write a critique..." class="w-full bg-transparent border-b border-zinc-800 py-2 text-sm text-zinc-300 focus:border-amber-500 focus:outline-none transition-colors">
                            <button class="absolute right-0 bottom-2 text-xs text-amber-500 font-cinema hover:text-white">PUBLISH</button>
                        </div>
                    </div>
                } @else {
                    <div class="p-4 bg-zinc-900/30 border border-zinc-800 text-center rounded">
                        <span class="font-script text-zinc-500 text-xs">Purchase a ticket (Sign In) to view Storyboards, Director Cuts, and leave reviews.</span>
                    </div>
                }

            </article>
            <div class="w-full h-px bg-gradient-to-r from-transparent via-zinc-800 to-transparent"></div>
          }
        }
      </div>

      <!-- Cinema Mode Overlay -->
      @if (activeCinemaPost(); as post) {
        <div class="fixed inset-0 z-50 bg-black flex flex-col" (click)="closeCinemaMode()">
           
           <!-- Cinema Screen Area -->
           <div class="flex-1 flex items-center justify-center p-8 overflow-hidden relative">
              
              <!-- Video constrained by viewport -->
              @if(post.videoUrl) {
                <video 
                    [src]="post.videoUrl" 
                    loop muted autoplay playsinline 
                    class="max-w-full max-h-full shadow-2xl shadow-black/50"
                    (click)="$event.stopPropagation()"> <!-- Clicking video shouldn't close it immediately, maybe toggle pause? leaving empty for now -->
                </video>
              }

              <!-- Close Button -->
              <button (click)="closeCinemaMode()" class="absolute top-8 right-8 text-white/50 hover:text-white z-50">
                <span class="font-cinema text-xl">EXIT THEATER ✕</span>
              </button>
           </div>

           <!-- Subtitle Area (Black bar at bottom) -->
           <!-- Per request: Black background, subtitles moved here so they don't cover video -->
           <div class="h-32 bg-black border-t border-zinc-900 flex items-center justify-center text-center px-12 shrink-0">
              <p class="text-amber-100 font-serif text-2xl tracking-wide leading-relaxed">
                 {{ post.subtitles?.[1]?.text || "..." }}
              </p>
           </div>

        </div>
      }

    </div>
  `,
  styles: [`
    .animate-fade-in {
        animation: fadeIn 1s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
  `]
})
export class PremiereComponent {
  vellumService = inject(VellumiumService);
  activeCinemaPost = signal<Post | null>(null);

  openCinemaMode(post: Post) {
    this.activeCinemaPost.set(post);
  }

  closeCinemaMode() {
    this.activeCinemaPost.set(null);
  }
}
