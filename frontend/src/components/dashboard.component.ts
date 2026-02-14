
import { Component, inject, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { VellumiumService, Project } from '../services/vellumium.service';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="h-screen flex bg-[var(--cinema-bg)] text-[var(--cinema-text)] font-sans overflow-hidden selection:bg-[var(--cinema-primary)] selection:text-white">

      <!-- Sidebar -->
      <aside class="w-64 flex-shrink-0 flex flex-col bg-[var(--cinema-surface)] border-r border-[var(--cinema-border)] h-full z-30">
        
        <!-- Logo -->
        <div class="p-6">
          <div class="flex gap-2 items-center cursor-pointer hover:opacity-80 transition-opacity" (click)="vellumService.navigateTo('LANDING')">
            <img src="assets/logo/dashboard_logo.png" alt="Vellumium" class="h-10 w-auto object-contain select-none" draggable="false" />
            <img src="assets/logo/dashboard_text_logo.png" alt="Vellumium Text" class="h-4 w-auto object-contain mt-1 select-none" draggable="false" />
          </div>
        </div>

        <!-- Navigation -->
        <nav class="flex-1 flex flex-col gap-1 px-3 py-4 overflow-y-auto">
          <a class="flex items-center gap-3 py-2.5 px-4 rounded-lg bg-[var(--cinema-primary)]/10 text-[var(--cinema-primary)] font-medium cursor-pointer transition-colors shadow-sm shadow-[var(--cinema-primary)]/5">
            <span class="material-symbols-outlined text-[20px]">movie</span>
            <span class="text-sm">My Projects</span>
          </a>
          <a class="flex items-center gap-3 py-2.5 px-4 rounded-lg text-[var(--cinema-text-muted)] hover:bg-white/5 hover:text-white cursor-pointer transition-colors" (click)="vellumService.navigateTo('PREMIERE')">
            <span class="material-symbols-outlined text-[20px]">public</span>
            <span class="text-sm">Premiere</span>
          </a>
          <a class="flex items-center gap-3 py-2.5 px-4 rounded-lg text-[var(--cinema-text-muted)] hover:bg-white/5 hover:text-white cursor-pointer transition-colors">
            <span class="material-symbols-outlined text-[20px]">settings</span>
            <span class="text-sm">Settings</span>
          </a>
        </nav>

        <!-- User Profile -->
        <div class="p-4 border-t border-[var(--cinema-border)]">
          <div class="flex items-center gap-3 mb-3">
            <div class="w-9 h-9 rounded-full bg-gradient-to-br from-[var(--cinema-primary)] to-[var(--cinema-accent)] flex items-center justify-center flex-shrink-0 shadow-lg shadow-[var(--cinema-primary)]/20">
              <span class="font-bold text-white text-xs">{{ getUserInitials() }}</span>
            </div>
            <div class="flex-1 min-w-0">
              <p class="text-white text-sm font-medium truncate">{{ getUserEmail() }}</p>
              <p class="text-[var(--cinema-text-dim)] text-xs truncate">Free Plan</p>
            </div>
          </div>
          <button (click)="onLogout()" class="w-full py-2 text-xs font-medium text-[var(--cinema-text-muted)] hover:text-white hover:bg-white/5 rounded-lg transition-colors flex items-center justify-center gap-2">
            <span class="material-symbols-outlined text-[14px]">logout</span>
            Sign Out
          </button>
        </div>
      </aside>

      <!-- Main Content -->
      <main class="flex-1 flex flex-col min-w-0 relative">
        <!-- Background Gradients -->
        <div class="absolute inset-0 pointer-events-none overflow-hidden">
           <div class="absolute top-[-20%] right-[-10%] w-[600px] h-[600px] bg-[var(--cinema-primary)] opacity-[0.03] blur-[100px] rounded-full"></div>
        </div>

        <!-- Header -->
        <header class="flex-shrink-0 px-8 py-6 z-10">
          <div class="flex items-center justify-between">
            <div>
              <h2 class="text-2xl font-bold tracking-tight">Recent Projects</h2>
              <p class="text-[var(--cinema-text-muted)] text-sm mt-1">
                @if (!vellumService.loading() && vellumService.projects().length > 0) {
                  {{ vellumService.projects().length }} project{{ vellumService.projects().length === 1 ? '' : 's' }} showing
                } @else {
                  Manage and create your AI videos
                }
              </p>
            </div>
            <button (click)="showNewProjectModal.set(true)" class="px-5 py-2.5 bg-[var(--cinema-primary)] hover:bg-[#6015c7] text-white rounded-lg font-medium shadow-lg shadow-[var(--cinema-primary-dim)] transition-all flex items-center gap-2">
              <span class="material-symbols-outlined text-[20px]">add</span>
              <span>New Project</span>
            </button>
          </div>
        </header>

        <!-- Content Area -->
        <div class="flex-1 overflow-y-auto px-8 pb-8 z-10">

          <!-- Loading State -->
          @if (vellumService.loading()) {
            <div class="flex justify-center py-20">
              <div class="w-8 h-8 border-2 border-[var(--cinema-primary)] border-t-transparent rounded-full animate-spin"></div>
            </div>
          }

          <!-- Error State -->
          @if (vellumService.error()) {
            <div class="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm flex items-center gap-2">
              <span class="material-symbols-outlined text-[18px]">error</span>
              {{ vellumService.error() }}
            </div>
          }

          <!-- Empty State -->
          @if (!vellumService.loading() && vellumService.projects().length === 0) {
            <div class="flex flex-col items-center justify-center py-24 text-center">
              <div class="w-16 h-16 rounded-2xl bg-[var(--cinema-surface)] border border-[var(--cinema-border)] flex items-center justify-center mb-6 shadow-xl">
                 <span class="material-symbols-outlined text-3xl text-[var(--cinema-text-muted)]">movie_edit</span>
              </div>
              <h3 class="text-lg font-medium mb-2">No projects yet</h3>
              <p class="text-[var(--cinema-text-muted)] text-sm mb-6 max-w-xs">Start your first AI video production project to see it here.</p>
              <button (click)="showNewProjectModal.set(true)" class="px-5 py-2 text-sm bg-[var(--cinema-surface)] hover:bg-[var(--cinema-elevated)] border border-[var(--cinema-border)] rounded-lg transition-all">
                Create Project
              </button>
            </div>
          }

          <!-- Project Grid -->
          @if (!vellumService.loading() && vellumService.projects().length > 0) {
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              @for (project of vellumService.projects(); track project.id) {
                <div
                  class="group cursor-pointer bg-[var(--cinema-surface)] border border-[var(--cinema-border)] hover:border-[var(--cinema-primary)]/50 rounded-xl overflow-hidden transition-all duration-300 hover:shadow-2xl hover:shadow-[var(--cinema-primary-dim)] hover:-translate-y-1"
                  (click)="vellumService.selectProject(project)">
                  
                  <!-- Thumbnail -->
                   <div class="aspect-video relative bg-[#050510] overflow-hidden">
                    @if (project.thumbnail_url || getFirstSceneThumbnail(project)) {
                      <img
                        [src]="project.thumbnail_url || getFirstSceneThumbnail(project)"
                        [alt]="project.title"
                        class="absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" />
                      <div class="absolute inset-0 bg-black/20 group-hover:bg-black/0 transition-colors"></div>
                    } @else {
                       <!-- Abstract Pattern Generation based on Title -->
                      <div class="absolute inset-0 opacity-30" [style]="'background-image: ' + getProjectPattern(project.title)"></div>
                      <div class="absolute inset-0 flex items-center justify-center">
                         <span class="text-4xl font-bold opacity-20 text-white">{{ project.title.charAt(0).toUpperCase() }}</span>
                      </div>
                    }
                    
                    <!-- Hover Overlay -->
                    <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-[1px]">
                       <div class="bg-white/10 backdrop-blur-md rounded-full px-4 py-2 text-xs font-bold text-white border border-white/20 transform translate-y-2 group-hover:translate-y-0 transition-transform">
                          OPEN STUDIO
                       </div>
                    </div>
                  </div>

                  <!-- Info -->
                  <div class="p-4">
                    <h3 class="font-bold text-base mb-1 truncate group-hover:text-[var(--cinema-primary)] transition-colors">{{ project.title }}</h3>
                    <div class="flex items-center justify-between mt-3">
                       <span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide bg-[var(--cinema-elevated)] text-[var(--cinema-text-muted)] border border-[var(--cinema-border)]">
                         {{ project.genre || 'General' }}
                       </span>
                       <span class="text-[10px] text-[var(--cinema-text-dim)]">{{ project.created_at | date:'MMM d' }}</span>
                    </div>
                  </div>
                </div>
              }
            </div>
          }
        </div>
      </main>

      <!-- New Project Modal -->
      @if (showNewProjectModal()) {
        <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in" (click)="showNewProjectModal.set(false)">
          <div class="w-[480px] glass-panel rounded-xl shadow-2xl p-6 border border-[var(--cinema-border)] animate-scale-in" (click)="$event.stopPropagation()">

            <div class="flex items-center justify-between mb-6">
              <h3 class="text-lg font-bold">New Project</h3>
              <button (click)="showNewProjectModal.set(false)" class="text-[var(--cinema-text-muted)] hover:text-white transition-colors">
                <span class="material-symbols-outlined">close</span>
              </button>
            </div>

            <div class="space-y-4">
              <div>
                <label class="text-xs font-medium text-[var(--cinema-text-muted)] mb-1.5 block">Project Title</label>
                <input
                  type="text"
                  [(ngModel)]="newProjectTitle"
                  placeholder="e.g. Cyberpunk Nexus..."
                  class="w-full px-4 py-2.5 bg-[var(--cinema-bg)] border border-[var(--cinema-border)] rounded-lg text-white text-sm focus:border-[var(--cinema-primary)] focus:outline-none transition-colors placeholder-[var(--cinema-text-dim)]"
                />
              </div>

              <div>
                <label class="text-xs font-medium text-[var(--cinema-text-muted)] mb-1.5 block">Description</label>
                <textarea
                  [(ngModel)]="newProjectDescription"
                  placeholder="Brief synopsis..."
                  rows="3"
                  class="w-full px-4 py-2.5 bg-[var(--cinema-bg)] border border-[var(--cinema-border)] rounded-lg text-white text-sm focus:border-[var(--cinema-primary)] focus:outline-none transition-colors placeholder-[var(--cinema-text-dim)] resize-none"
                ></textarea>
              </div>

              <div>
                <label class="text-xs font-medium text-[var(--cinema-text-muted)] mb-1.5 block">Genre</label>
                <select
                  [(ngModel)]="newProjectGenre"
                  class="w-full px-4 py-2.5 bg-[var(--cinema-bg)] border border-[var(--cinema-border)] rounded-lg text-white text-sm focus:border-[var(--cinema-primary)] focus:outline-none transition-colors appearance-none cursor-pointer">
                  <option value="" class="bg-[var(--cinema-bg)]">Select genre...</option>
                  @for (g of genreOptions; track g) {
                    <option [value]="g" class="bg-[var(--cinema-bg)]">{{ g }}</option>
                  }
                </select>
              </div>

              <div class="pt-4">
                <button
                  (click)="onCreateProject()"
                  [disabled]="!newProjectTitle.trim()"
                  class="w-full py-2.5 bg-[var(--cinema-primary)] hover:bg-[#6015c7] text-white font-bold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-[var(--cinema-primary)]/20">
                  Create Project
                </button>
              </div>
            </div>
          </div>
        </div>
      }

    </div>
  `,
  styles: [`
    .animate-scale-in {
      animation: scaleIn 0.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .animate-fade-in {
      animation: fadeIn 0.2s ease-out forwards;
    }
    @keyframes scaleIn {
      from { transform: scale(0.95); opacity: 0; }
      to { transform: scale(1); opacity: 1; }
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
  `]
})
export class DashboardComponent implements OnInit {
  vellumService = inject(VellumiumService);
  authService = inject(AuthService);
  showNewProjectModal = signal(false);

  newProjectTitle = '';
  newProjectDescription = '';
  newProjectGenre = '';

  genreOptions = ['General', 'Sci-Fi', 'Horror', 'Drama', 'Comedy', 'Action', 'Thriller', 'Fantasy', 'Romance', 'Documentary', 'Animation', 'Noir', 'Western', 'Abstract'];

  getProjectPattern(title: string): string {
    // Generate a deterministic abstract gradient pattern based on the title
    const hash = title.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    const hue1 = hash % 360;
    const hue2 = (hash + 180) % 360;
    return `linear-gradient(135deg, hsla(${hue1}, 70%, 50%, 0.1) 0%, hsla(${hue2}, 70%, 50%, 0.1) 100%)`;
  }

  ngOnInit() {
    this.vellumService.loadProjects();
  }

  getFirstSceneThumbnail(project: Project): string | null {
    if (project.scenes.length > 0 && project.scenes[0].thumbnail_url) {
      return project.scenes[0].thumbnail_url;
    }
    return null;
  }

  getUserInitials(): string {
    const user = this.authService.currentUser();
    if (!user?.email) return '?';
    return user.email.charAt(0).toUpperCase();
  }

  getUserEmail(): string {
    const user = this.authService.currentUser();
    return user?.email || '';
  }

  async onCreateProject() {
    if (!this.newProjectTitle.trim()) return;

    const project = await this.vellumService.createProject(
      this.newProjectTitle.trim(),
      this.newProjectDescription.trim(),
      this.newProjectGenre.trim() || 'General'
    );

    if (project) {
      this.showNewProjectModal.set(false);
      this.newProjectTitle = '';
      this.newProjectDescription = '';
      this.newProjectGenre = '';
    }
  }

  async onLogout() {
    await this.authService.signOut();
    this.vellumService.navigateTo('LANDING');
  }
}
