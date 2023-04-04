import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WordReassignComponentComponent } from './word-reassign-component.component';

describe('WordReassignComponentComponent', () => {
  let component: WordReassignComponentComponent;
  let fixture: ComponentFixture<WordReassignComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ WordReassignComponentComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(WordReassignComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
