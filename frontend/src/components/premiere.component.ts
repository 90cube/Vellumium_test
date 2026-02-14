
import { Component, inject, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VellumiumService, Post } from '../services/vellumium.service';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-premiere',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="min-h-screen bg-[var(--cinema-bg)] text-[var(--cinema-text)] pb-20 font-display projector-beam">

      <!-- Theater Nav -->
      <nav class="sticky top-0 z-40 cinema-panel border-b border-[var(--cinema-border-gold)] px-6 py-3 flex justify-between items-center">
        <div class="flex items-center gap-4">
          <button (click)="vellumService.backToDashboard()" class="btn-cinema flex items-center justify-center rounded-lg size-10 text-[var(--cinema-gold)] hover:bg-[var(--cinema-gold)]/10 transition-colors">
            <span class="material-symbols-outlined">arrow_back</span>
          </button>
          <img src="assets/logo/big-logo.png" alt="Vellumium" class="h-8 cursor-pointer hover:opacity-80 transition-opacity" (click)="vellumService.navigateTo('LANDING')" />
          <div class="flex flex-col">
            <h1 class="font-cinema text-[var(--cinema-gold)] text-lg tracking-[0.15em]">PREMIERE</h1>
            <div class="marquee-border mt-0.5 w-20 h-[2px]"></div>
          </div>
          <div class="h-6 w-px bg-[var(--cinema-border)]"></div>
          <!-- Central Tabs -->
          <div class="hidden md:flex bg-[var(--cinema-surface)] rounded-lg p-1 border border-[var(--cinema-border)]">
            <button class="px-4 py-1.5 rounded-md bg-[var(--cinema-gold)]/15 text-[var(--cinema-gold)] text-sm font-medium border border-[var(--cinema-gold)]/20">Featured</button>
            <button class="px-4 py-1.5 rounded-md text-[var(--cinema-text-muted)] hover:text-[var(--cinema-gold)] text-sm font-medium transition-colors">Noir</button>
            <button class="px-4 py-1.5 rounded-md text-[var(--cinema-text-muted)] hover:text-[var(--cinema-gold)] text-sm font-medium transition-colors">Sci-Fi</button>
            <button class="px-4 py-1.5 rounded-md text-[var(--cinema-text-muted)] hover:text-[var(--cinema-gold)] text-sm font-medium transition-colors">Experimental</button>
          </div>
        </div>
        <div class="flex items-center gap-4">
          @if(authService.isLoggedIn()) {
            <div class="flex items-center gap-2">
              <div class="w-8 h-8 rounded-full border-2 border-[var(--cinema-gold)]/30 bg-[var(--cinema-gold)]/10 flex items-center justify-center text-xs font-cinema text-[var(--cinema-gold)] font-bold">
                {{ (authService.currentUser()?.email || '?')[0].toUpperCase() }}
              </div>
            </div>
          }
        </div>
      </nav>

      <!-- Loading -->
      @if (vellumService.loading()) {
        <div class="flex justify-center py-20">
          <div class="w-10 h-10 border-3 border-[var(--cinema-gold)] border-t-transparent rounded-full animate-spin"></div>
        </div>
      }

      <!-- Empty State -->
      @if (!vellumService.loading() && vellumService.communityPosts().length === 0) {
        <div class="max-w-4xl mx-auto text-center py-20">
          <span class="material-symbols-outlined text-7xl text-[var(--cinema-gold)]/30 mb-6 block">theaters</span>
          <p class="font-cinema text-[var(--cinema-gold)]/50 text-xl mb-3">NO SCREENINGS YET</p>
          <p class="text-[var(--cinema-text-muted)] text-sm font-body">The theater is empty. Be the first to premiere your work.</p>
        </div>
      }

      <!-- Feed -->
      <div class="max-w-4xl mx-auto mt-12 space-y-16 px-4">

        @for (post of vellumService.communityPosts(); track post.id) {

          @if (authService.isLoggedIn() || (!post.show_storyboard && !post.show_director_cut)) {

            <article class="flex flex-col gap-6 animate-fade-in">

              <!-- Header -->
              <div class="flex justify-between items-end border-b border-[var(--cinema-border-gold)] pb-3">
                <div>
                  <h2 class="font-cinema text-[var(--cinema-gold-bright)] text-3xl tracking-wide">{{ post.title }}</h2>
                  <div class="flex gap-3 mt-2 items-center">
                    <span class="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-[var(--cinema-gold)]/10 text-[var(--cinema-gold)] border border-[var(--cinema-gold)]/20 tracking-wider uppercase">{{ post.genre || 'General' }}</span>
                    <span class="text-[var(--cinema-text-muted)] text-xs font-body">Dir. {{ post.author_name }}</span>
                  </div>
                </div>
                <div class="flex gap-2">
                  <button class="btn-cinema p-2 rounded-lg" title="Share">
                    <span class="material-symbols-outlined text-[20px]">ios_share</span>
                  </button>
                </div>
              </div>

              <!-- Content Viewer (Brass-framed screening) -->
              <div class="w-full bg-black shadow-2xl relative group rounded-xl overflow-hidden brass-frame">

                @if (post.video_url) {
                  <div
                    class="aspect-video relative bg-black flex flex-col justify-center cursor-pointer group/video"
                    (click)="openCinemaMode(post)"
                  >
                    <div class="relative flex-1 overflow-hidden">
                      <video
                        [src]="post.video_url"
                        loop muted autoplay playsinline
                        class="w-full h-full object-cover">
                      </video>
                    </div>
                    <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover/video:opacity-100 transition-opacity bg-black/40">
                      <button class="bg-[var(--cinema-gold)]/15 backdrop-blur-md border border-[var(--cinema-gold)]/30 rounded-full p-4 hover:bg-[var(--cinema-gold)]/25 text-[var(--cinema-gold)] cinema-glow transition-all">
                        <span class="material-symbols-outlined text-4xl">play_arrow</span>
                      </button>
                    </div>
                  </div>

                } @else if (post.thumbnail_url) {
                  <div class="aspect-video bg-[var(--cinema-bg)] relative overflow-hidden group/sb spotlight-hover">
                    <img [src]="post.thumbnail_url" class="w-full h-full object-contain transition-transform duration-700 group-hover/sb:scale-105 origin-center">
                  </div>

                } @else {
                  <div class="aspect-video bg-[var(--cinema-bg)] flex items-center justify-center">
                    <span class="material-symbols-outlined text-5xl text-[var(--cinema-gold)]/20">movie</span>
                  </div>
                }

              </div>

              <!-- Footer / Social -->
              @if (authService.isLoggedIn()) {
                <div class="flex flex-col gap-4 pl-4 border-l-2 border-[var(--cinema-gold)]/20">
                  <div class="flex gap-6 items-center">
                    <button
                      (click)="onToggleReaction(post.id, 'like')"
                      class="flex items-center gap-2 transition-colors group"
                      [class.text-[var(--cinema-red-bright)]]="post.user_reaction === 'like'"
                      [class.text-[var(--cinema-text-muted)]]="post.user_reaction !== 'like'"
                    >
                      <span class="material-symbols-outlined group-hover:scale-125 transition-transform" [class.fill-1]="post.user_reaction === 'like'">favorite</span>
                      <span class="font-mono text-xs">{{ post.like_count }}</span>
                    </button>
                    <button
                      (click)="onToggleReaction(post.id, 'dislike')"
                      class="flex items-center gap-2 transition-colors group"
                      [class.text-[var(--cinema-text-dim)]]="post.user_reaction === 'dislike'"
                      [class.text-[var(--cinema-text-muted)]]="post.user_reaction !== 'dislike'"
                    >
                      <span class="material-symbols-outlined group-hover:scale-125 transition-transform" [class.fill-1]="post.user_reaction === 'dislike'">thumb_down</span>
                      <span class="font-mono text-xs">{{ post.dislike_count }}</span>
                    </button>
                    <div class="flex items-center gap-2 text-[var(--cinema-text-muted)]">
                      <span class="material-symbols-outlined text-[16px]">chat_bubble</span>
                      <span class="font-mono text-xs">{{ post.comment_count }} Critiques</span>
                    </div>
                  </div>

                  <!-- Comments -->
                  @if (post.comments.length > 0) {
                    <div class="space-y-2 ml-2">
                      @for (comment of post.comments; track comment.id) {
                        <div class="flex gap-2 text-xs font-body">
                          <span class="text-[var(--cinema-gold)] font-bold shrink-0">{{ comment.author_name }}</span>
                          <span class="text-[var(--cinema-text-muted)]">{{ comment.text }}</span>
                        </div>
                      }
                    </div>
                  }

                  <!-- Comment Input -->
                  <div class="relative">
                    <input
                      type="text"
                      placeholder="Write a critique..."
                      [(ngModel)]="commentTexts[post.id]"
                      (keydown.enter)="onAddComment(post.id)"
                      class="w-full bg-transparent border-b border-[var(--cinema-border)] py-2 text-sm text-[var(--cinema-text)] focus:border-[var(--cinema-gold)]/50 focus:outline-none transition-colors font-body placeholder-[var(--cinema-text-dim)]">
                    <button
                      (click)="onAddComment(post.id)"
                      class="absolute right-0 bottom-2 text-xs text-[var(--cinema-gold)] font-bold font-cinema tracking-wider hover:text-[var(--cinema-gold-bright)] transition-colors">PUBLISH</button>
                  </div>
                </div>
              } @else {
                <div class="p-4 cinema-panel rounded-lg text-center">
                  <span class="text-[var(--cinema-text-muted)] text-xs font-body">Sign in to view Storyboards, Director Cuts, and leave reviews.</span>
                </div>
              }

            </article>
            <div class="gold-line my-2"></div>
          }
        }
      </div>

      <!-- Cinema Mode Overlay (Private Screening Room) -->
      @if (activeCinemaPost(); as post) {
        <div class="fixed inset-0 z-50 bg-black flex flex-col" (click)="closeCinemaMode()">

          <!-- Projector beam effect over the screen -->
          <div class="flex-1 flex items-center justify-center p-8 overflow-hidden relative projector-beam">

            @if(post.video_url) {
              <div class="brass-frame rounded-lg cinema-glow-strong" (click)="$event.stopPropagation()">
                <video
                  [src]="post.video_url"
                  loop muted autoplay playsinline
                  class="max-w-full max-h-[75vh] block">
                </video>
              </div>
            }

            <button (click)="closeCinemaMode()" class="btn-cinema absolute top-8 right-8 z-50 flex items-center gap-2 px-4 py-2 rounded-lg">
              <span class="material-symbols-outlined text-sm">close</span>
              <span class="text-sm font-cinema tracking-wider">EXIT</span>
            </button>
          </div>

          <!-- Bottom bar: title card -->
          <div class="h-24 bg-[var(--cinema-panel)] border-t border-[var(--cinema-border-gold)] flex items-center justify-center text-center px-12 shrink-0">
            <div class="flex flex-col items-center gap-1">
              <p class="font-cinema text-[var(--cinema-gold-bright)] text-2xl tracking-[0.1em]">
                {{ post.title }}
              </p>
              <div class="marquee-border w-32 h-[2px]"></div>
            </div>
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
export class PremiereComponent implements OnInit {
  vellumService = inject(VellumiumService);
  authService = inject(AuthService);
  activeCinemaPost = signal<Post | null>(null);
  commentTexts: Record<string, string> = {};

  ngOnInit() {
    this.vellumService.loadCommunityPosts();
  }

  openCinemaMode(post: Post) {
    this.activeCinemaPost.set(post);
  }

  closeCinemaMode() {
    this.activeCinemaPost.set(null);
  }

  async onToggleReaction(postId: string, type: 'like' | 'dislike') {
    await this.vellumService.toggleReaction(postId, type);
  }

  async onAddComment(postId: string) {
    const text = this.commentTexts[postId]?.trim();
    if (!text) return;

    const success = await this.vellumService.addComment(postId, text);
    if (success) {
      this.commentTexts[postId] = '';
    }
  }
}
