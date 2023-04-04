import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WordCloudComponentComponent } from './word-cloud-component.component';

describe('WordCloudComponentComponent', () => {
  let component: WordCloudComponentComponent;
  let fixture: ComponentFixture<WordCloudComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ WordCloudComponentComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(WordCloudComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
