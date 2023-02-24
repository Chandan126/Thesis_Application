import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ArticleExplanationComponent } from './article-explanation.component';

describe('ArticleExplanationComponent', () => {
  let component: ArticleExplanationComponent;
  let fixture: ComponentFixture<ArticleExplanationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ArticleExplanationComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ArticleExplanationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
