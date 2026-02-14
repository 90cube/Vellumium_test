
import { Injectable, signal, inject } from '@angular/core';
import { SupabaseService } from './supabase.service';
import { AuthService } from './auth.service';

export type ViewState = 'LANDING' | 'DASHBOARD' | 'EDITOR' | 'STORYBOARD_CANVAS' | 'DIRECTORSCUT_VIDEOGEN' | 'PREMIERE';

// Project stages: functional layers within every project (not project types)
// Storyboard: image editing per scene (scene → canvas → layers)
// Director's Cut: i2v conversion per scene (confirmed images → video)
// Film: timeline editing of all confirmed videos stitched together
export type ProjectStage = 'STORYBOARD' | 'DIRECTORS_CUT' | 'FILM';

export interface Layer {
  id: string;
  type: 'image' | 'draw' | 'text' | 'mask';
  content_url: string | null;
  caption: string | null;
  z_index: number;
  locked: boolean;
  scene_id: string;
  name: string;
  opacity: number;
  blend_mode: string;
  visible: boolean;
}

export interface Scene {
  id: string;
  title: string;
  description: string | null;
  thumbnail_url: string | null;          // Storyboard: auto-set to latest edited image
  confirmed_image_url: string | null;    // Finalized image from Storyboard (passed to Director's Cut)
  generated_video_url: string | null;    // Director's Cut: i2v result video
  video_thumbnail_url: string | null;    // Director's Cut: first frame of generated video
  sort_order: number;
  clip_start_ms: number;                 // Film: clip trim start
  clip_end_ms: number | null;            // Film: clip trim end
  clip_trimmed: boolean;                 // Film: whether clip was trimmed
  project_id: string;
  character_id: string | null;           // Bound character from library
  location_id: string | null;            // Bound location from library
  layers: Layer[];
}

export interface Project {
  id: string;
  title: string;
  description: string | null;
  genre: string | null;
  thumbnail_url: string | null;
  style_preset_id: string | null;
  created_at: string;
  user_id: string;
  scenes: Scene[];
}

export interface Post {
  id: string;
  project_id: string;
  user_id: string;
  title: string;
  genre: string | null;
  video_url: string | null;
  thumbnail_url: string | null;
  show_storyboard: boolean;
  show_director_cut: boolean;
  created_at: string;
  // Joined data
  author_name: string;
  comment_count: number;
  like_count: number;
  dislike_count: number;
  user_reaction: 'like' | 'dislike' | null;
  comments: PostComment[];
}

export interface PostComment {
  id: string;
  user_id: string;
  text: string;
  created_at: string;
  author_name: string;
}

@Injectable({
  providedIn: 'root'
})
export class VellumiumService {
  private supabaseService = inject(SupabaseService);
  private authService = inject(AuthService);
  private supabase = this.supabaseService.supabase;

  // Navigation State
  readonly currentView = signal<ViewState>('LANDING');
  readonly selectedProject = signal<Project | null>(null);
  readonly selectedScene = signal<Scene | null>(null);

  // Project Stage: which functional layer the user is working in
  readonly currentStage = signal<ProjectStage>('STORYBOARD');

  // Data Signals
  readonly projects = signal<Project[]>([]);
  readonly communityPosts = signal<Post[]>([]);
  readonly sceneLayers = signal<Layer[]>([]);

  // UI State
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);

  // Transition State: 'idle' | 'closing' | 'opening'
  readonly transitioning = signal<'idle' | 'closing' | 'opening'>('idle');
  private transitionBusy = false;

  // --- Navigation with transition ---
  navigateTo(view: ViewState, skipTransition = false) {
    if (skipTransition) {
      this.currentView.set(view);
      return;
    }
    this.transitionTo(() => {
      this.currentView.set(view);
    });
  }

  selectProject(project: Project) {
    this.transitionTo(() => {
      this.selectedProject.set(project);
      this.currentStage.set('STORYBOARD');
      this.currentView.set('EDITOR');
    });
  }

  // Switch stage within the current project (no view transition needed)
  switchStage(stage: ProjectStage) {
    this.currentStage.set(stage);
    // Clear scene selection when switching stages
    this.selectedScene.set(null);
    this.sceneLayers.set([]);
  }

  // Storyboard: click scene → open storyboard canvas for image editing
  selectScene(scene: Scene) {
    this.transitionTo(() => {
      this.selectedScene.set(scene);
      this.currentView.set('STORYBOARD_CANVAS');
      this.loadLayers(scene.id);
    });
  }

  // Director's Cut: click scene → open videogen workflow
  selectSceneForVideogen(scene: Scene) {
    this.transitionTo(() => {
      this.selectedScene.set(scene);
      this.currentView.set('DIRECTORSCUT_VIDEOGEN');
    });
  }

  backToEditor() {
    this.transitionTo(() => {
      this.selectedScene.set(null);
      this.sceneLayers.set([]);
      this.currentView.set('EDITOR');
    });
  }

  backToDashboard() {
    this.transitionTo(() => {
      this.selectedProject.set(null);
      this.selectedScene.set(null);
      this.sceneLayers.set([]);
      this.currentStage.set('STORYBOARD');
      this.currentView.set('DASHBOARD');
    });
  }

  private transitionTo(swap: () => void) {
    if (this.transitionBusy) return;
    this.transitionBusy = true;

    // Phase 1: curtains close
    this.transitioning.set('closing');
    setTimeout(() => {
      // Phase 2: swap content while curtains cover screen
      swap();
      // Phase 3: curtains open
      this.transitioning.set('opening');
      setTimeout(() => {
        this.transitioning.set('idle');
        this.transitionBusy = false;
      }, 700);
    }, 700);
  }

  // --- Projects ---
  async loadProjects(): Promise<void> {
    const user = this.authService.currentUser();
    if (!user) return;

    this.loading.set(true);
    this.error.set(null);
    try {
      const { data: projectRows, error: projErr } = await this.supabase
        .from('projects')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (projErr) throw projErr;

      const projectIds = (projectRows || []).map((p: any) => p.id);
      let scenesMap: Record<string, Scene[]> = {};

      if (projectIds.length > 0) {
        const { data: sceneRows, error: sceneErr } = await this.supabase
          .from('scenes')
          .select('*')
          .in('project_id', projectIds)
          .order('sort_order', { ascending: true });

        if (sceneErr) throw sceneErr;

        for (const s of (sceneRows || [])) {
          const scene: Scene = {
            id: s.id,
            title: s.title,
            description: s.description,
            thumbnail_url: s.thumbnail_url,
            confirmed_image_url: s.confirmed_image_url,
            generated_video_url: s.generated_video_url,
            video_thumbnail_url: s.video_thumbnail_url,
            sort_order: s.sort_order,
            clip_start_ms: s.clip_start_ms ?? 0,
            clip_end_ms: s.clip_end_ms,
            clip_trimmed: s.clip_trimmed ?? false,
            project_id: s.project_id,
            character_id: s.character_id ?? null,
            location_id: s.location_id ?? null,
            layers: []
          };
          if (!scenesMap[s.project_id]) scenesMap[s.project_id] = [];
          scenesMap[s.project_id].push(scene);
        }
      }

      const projects: Project[] = (projectRows || []).map((p: any) => ({
        id: p.id,
        title: p.title,
        description: p.description,
        genre: p.genre,
        thumbnail_url: p.thumbnail_url,
        style_preset_id: p.style_preset_id ?? null,
        created_at: p.created_at,
        user_id: p.user_id,
        scenes: scenesMap[p.id] || []
      }));

      this.projects.set(projects);
    } catch (e: any) {
      console.error('loadProjects error:', e);
      this.error.set(e.message || 'Failed to load projects');
    } finally {
      this.loading.set(false);
    }
  }

  async createProject(title: string, description: string, genre: string): Promise<Project | null> {
    const user = this.authService.currentUser();
    if (!user) return null;

    this.error.set(null);
    try {
      const { data, error } = await this.supabase
        .from('projects')
        .insert({ title, description, genre, user_id: user.id })
        .select()
        .single();

      if (error) throw error;

      const project: Project = {
        id: data.id,
        title: data.title,
        description: data.description,
        genre: data.genre,
        thumbnail_url: data.thumbnail_url,
        style_preset_id: data.style_preset_id ?? null,
        created_at: data.created_at,
        user_id: data.user_id,
        scenes: []
      };

      this.projects.update(prev => [project, ...prev]);
      return project;
    } catch (e: any) {
      console.error('createProject error:', e);
      this.error.set(e.message || 'Failed to create project');
      return null;
    }
  }

  // --- Project Style ---
  async setProjectStyle(projectId: string, stylePresetId: string | null): Promise<boolean> {
    try {
      const { error } = await this.supabase
        .from('projects')
        .update({ style_preset_id: stylePresetId })
        .eq('id', projectId);
      if (error) throw error;

      const project = this.selectedProject();
      if (project?.id === projectId) {
        this.selectedProject.set({ ...project, style_preset_id: stylePresetId });
      }
      this.projects.update(prev =>
        prev.map(p => p.id === projectId ? { ...p, style_preset_id: stylePresetId } : p)
      );
      return true;
    } catch (e: any) {
      console.error('setProjectStyle error:', e);
      return false;
    }
  }

  // --- Scenes ---
  async createScene(projectId: string, title: string, description: string): Promise<Scene | null> {
    this.error.set(null);
    try {
      const currentProject = this.selectedProject();
      const maxOrder = currentProject?.scenes?.length
        ? Math.max(...currentProject.scenes.map(s => s.sort_order))
        : -1;

      const { data, error } = await this.supabase
        .from('scenes')
        .insert({
          project_id: projectId,
          title,
          description,
          sort_order: maxOrder + 1
        })
        .select()
        .single();

      if (error) throw error;

      const scene: Scene = {
        id: data.id,
        title: data.title,
        description: data.description,
        thumbnail_url: data.thumbnail_url,
        confirmed_image_url: data.confirmed_image_url,
        generated_video_url: data.generated_video_url,
        video_thumbnail_url: data.video_thumbnail_url,
        sort_order: data.sort_order,
        clip_start_ms: data.clip_start_ms ?? 0,
        clip_end_ms: data.clip_end_ms,
        clip_trimmed: data.clip_trimmed ?? false,
        project_id: data.project_id,
        character_id: data.character_id ?? null,
        location_id: data.location_id ?? null,
        layers: []
      };

      if (currentProject && currentProject.id === projectId) {
        const updated = { ...currentProject, scenes: [...currentProject.scenes, scene] };
        this.selectedProject.set(updated);
        this.projects.update(prev => prev.map(p => p.id === projectId ? updated : p));
      }

      return scene;
    } catch (e: any) {
      console.error('createScene error:', e);
      this.error.set(e.message || 'Failed to create scene');
      return null;
    }
  }

  // --- Layers ---
  async loadLayers(sceneId: string): Promise<void> {
    this.error.set(null);
    try {
      const { data, error } = await this.supabase
        .from('layers')
        .select('*')
        .eq('scene_id', sceneId)
        .order('z_index', { ascending: true });

      if (error) throw error;

      const layers: Layer[] = (data || []).map((l: any) => ({
        id: l.id,
        type: l.type,
        content_url: l.content_url,
        caption: l.caption,
        z_index: l.z_index,
        locked: l.locked,
        scene_id: l.scene_id,
        name: l.name ?? '',
        opacity: l.opacity ?? 1.0,
        blend_mode: l.blend_mode ?? 'source-over',
        visible: l.visible ?? true,
      }));

      this.sceneLayers.set(layers);
    } catch (e: any) {
      console.error('loadLayers error:', e);
      this.error.set(e.message || 'Failed to load layers');
    }
  }

  async createLayer(sceneId: string, type: 'image' | 'draw' | 'text' | 'mask'): Promise<Layer | null> {
    this.error.set(null);
    try {
      const currentLayers = this.sceneLayers();
      const maxZ = currentLayers.length > 0
        ? Math.max(...currentLayers.map(l => l.z_index))
        : -1;

      const { data, error } = await this.supabase
        .from('layers')
        .insert({
          scene_id: sceneId,
          type,
          z_index: maxZ + 1,
          locked: false
        })
        .select()
        .single();

      if (error) throw error;

      const layer: Layer = {
        id: data.id,
        type: data.type,
        content_url: data.content_url,
        caption: data.caption,
        z_index: data.z_index,
        locked: data.locked,
        scene_id: data.scene_id,
        name: data.name ?? '',
        opacity: data.opacity ?? 1.0,
        blend_mode: data.blend_mode ?? 'source-over',
        visible: data.visible ?? true,
      };

      this.sceneLayers.update(prev => [...prev, layer]);
      return layer;
    } catch (e: any) {
      console.error('createLayer error:', e);
      this.error.set(e.message || 'Failed to create layer');
      return null;
    }
  }

  // --- Scene Bindings (Character / Location) ---
  async bindCharacterToScene(sceneId: string, characterId: string | null): Promise<boolean> {
    try {
      const { error } = await this.supabase
        .from('scenes')
        .update({ character_id: characterId })
        .eq('id', sceneId);
      if (error) throw error;
      this.updateSceneLocally(sceneId, { character_id: characterId });
      return true;
    } catch (e: any) {
      console.error('bindCharacterToScene error:', e);
      return false;
    }
  }

  async bindLocationToScene(sceneId: string, locationId: string | null): Promise<boolean> {
    try {
      const { error } = await this.supabase
        .from('scenes')
        .update({ location_id: locationId })
        .eq('id', sceneId);
      if (error) throw error;
      this.updateSceneLocally(sceneId, { location_id: locationId });
      return true;
    } catch (e: any) {
      console.error('bindLocationToScene error:', e);
      return false;
    }
  }

  private updateSceneLocally(sceneId: string, updates: Partial<Scene>) {
    const project = this.selectedProject();
    if (project) {
      const updatedScenes = project.scenes.map(s =>
        s.id === sceneId ? { ...s, ...updates } : s
      );
      this.selectedProject.set({ ...project, scenes: updatedScenes });
    }
    const scene = this.selectedScene();
    if (scene?.id === sceneId) {
      this.selectedScene.set({ ...scene, ...updates });
    }
  }

  // --- Storyboard: confirm image for a scene ---
  async confirmSceneImage(sceneId: string, imageUrl: string): Promise<void> {
    this.error.set(null);
    try {
      const { error } = await this.supabase
        .from('scenes')
        .update({ confirmed_image_url: imageUrl })
        .eq('id', sceneId);

      if (error) throw error;

      // Update local state
      const project = this.selectedProject();
      if (project) {
        const updatedScenes = project.scenes.map(s =>
          s.id === sceneId ? { ...s, confirmed_image_url: imageUrl } : s
        );
        this.selectedProject.set({ ...project, scenes: updatedScenes });
      }
    } catch (e: any) {
      console.error('confirmSceneImage error:', e);
      this.error.set(e.message || 'Failed to confirm scene image');
    }
  }

  // --- Generation Queue ---
  async submitGenerationRequest(sceneId: string, layerId: string, prompt: string, params: Record<string, any> = {}): Promise<boolean> {
    const user = this.authService.currentUser();
    if (!user) return false;

    this.error.set(null);
    try {
      const { error } = await this.supabase
        .from('generation_queue')
        .insert({
          user_id: user.id,
          scene_id: sceneId,
          layer_id: layerId,
          prompt,
          params,
          status: 'pending'
        });

      if (error) throw error;
      return true;
    } catch (e: any) {
      console.error('submitGenerationRequest error:', e);
      this.error.set(e.message || 'Failed to submit generation request');
      return false;
    }
  }

  // --- Community Posts ---
  async loadCommunityPosts(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const user = this.authService.currentUser();

      const { data: postRows, error: postErr } = await this.supabase
        .from('posts')
        .select(`
          *,
          profiles!posts_user_id_fkey ( display_name )
        `)
        .order('created_at', { ascending: false });

      if (postErr) throw postErr;

      const posts: Post[] = [];
      for (const row of (postRows || [])) {
        const { count: commentCount } = await this.supabase
          .from('comments')
          .select('*', { count: 'exact', head: true })
          .eq('post_id', row.id);

        const { count: likeCount } = await this.supabase
          .from('reactions')
          .select('*', { count: 'exact', head: true })
          .eq('post_id', row.id)
          .eq('type', 'like');

        const { count: dislikeCount } = await this.supabase
          .from('reactions')
          .select('*', { count: 'exact', head: true })
          .eq('post_id', row.id)
          .eq('type', 'dislike');

        let userReaction: 'like' | 'dislike' | null = null;
        if (user) {
          const { data: reactionData } = await this.supabase
            .from('reactions')
            .select('type')
            .eq('post_id', row.id)
            .eq('user_id', user.id)
            .maybeSingle();
          userReaction = reactionData?.type ?? null;
        }

        const { data: commentRows } = await this.supabase
          .from('comments')
          .select(`
            *,
            profiles!comments_user_id_fkey ( display_name )
          `)
          .eq('post_id', row.id)
          .order('created_at', { ascending: true });

        const comments: PostComment[] = (commentRows || []).map((c: any) => ({
          id: c.id,
          user_id: c.user_id,
          text: c.text,
          created_at: c.created_at,
          author_name: c.profiles?.display_name || 'Anonymous'
        }));

        posts.push({
          id: row.id,
          project_id: row.project_id,
          user_id: row.user_id,
          title: row.title,
          genre: row.genre,
          video_url: row.video_url,
          thumbnail_url: row.thumbnail_url,
          show_storyboard: row.show_storyboard,
          show_director_cut: row.show_director_cut,
          created_at: row.created_at,
          author_name: row.profiles?.display_name || 'Anonymous',
          comment_count: commentCount || 0,
          like_count: likeCount || 0,
          dislike_count: dislikeCount || 0,
          user_reaction: userReaction,
          comments
        });
      }

      this.communityPosts.set(posts);
    } catch (e: any) {
      console.error('loadCommunityPosts error:', e);
      this.error.set(e.message || 'Failed to load posts');
    } finally {
      this.loading.set(false);
    }
  }

  async createPost(
    projectId: string,
    title: string,
    genre: string,
    options: { videoUrl?: string; thumbnailUrl?: string; showStoryboard?: boolean; showDirectorCut?: boolean } = {}
  ): Promise<boolean> {
    const user = this.authService.currentUser();
    if (!user) return false;

    this.error.set(null);
    try {
      const { error } = await this.supabase
        .from('posts')
        .insert({
          project_id: projectId,
          user_id: user.id,
          title,
          genre,
          video_url: options.videoUrl || null,
          thumbnail_url: options.thumbnailUrl || null,
          show_storyboard: options.showStoryboard ?? false,
          show_director_cut: options.showDirectorCut ?? false
        });

      if (error) throw error;

      await this.loadCommunityPosts();
      return true;
    } catch (e: any) {
      console.error('createPost error:', e);
      this.error.set(e.message || 'Failed to create post');
      return false;
    }
  }

  async addComment(postId: string, text: string): Promise<boolean> {
    const user = this.authService.currentUser();
    if (!user) return false;

    this.error.set(null);
    try {
      const { error } = await this.supabase
        .from('comments')
        .insert({ post_id: postId, user_id: user.id, text });

      if (error) throw error;

      await this.loadCommunityPosts();
      return true;
    } catch (e: any) {
      console.error('addComment error:', e);
      this.error.set(e.message || 'Failed to add comment');
      return false;
    }
  }

  async toggleReaction(postId: string, type: 'like' | 'dislike'): Promise<void> {
    const user = this.authService.currentUser();
    if (!user) return;

    this.error.set(null);
    try {
      const { data: existing } = await this.supabase
        .from('reactions')
        .select('id, type')
        .eq('post_id', postId)
        .eq('user_id', user.id)
        .maybeSingle();

      if (existing) {
        if (existing.type === type) {
          await this.supabase.from('reactions').delete().eq('id', existing.id);
        } else {
          await this.supabase.from('reactions').update({ type }).eq('id', existing.id);
        }
      } else {
        await this.supabase.from('reactions').insert({
          post_id: postId,
          user_id: user.id,
          type
        });
      }

      await this.loadCommunityPosts();
    } catch (e: any) {
      console.error('toggleReaction error:', e);
      this.error.set(e.message || 'Failed to toggle reaction');
    }
  }
}
