import { Pipe, PipeTransform } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Pipe({
  name: 'highlightWords'
})
export class HighlightWordsPipe implements PipeTransform {

  constructor(private sanitizer: DomSanitizer) {}

  transform(sentence: string, importantWords: string[]): SafeHtml {
    const regex = new RegExp(`(${importantWords.join('|')})`, 'gi');
    const highlightedSentence = sentence.replace(regex, '<span class="highlight">$1</span>');
    return this.sanitizer.bypassSecurityTrustHtml(highlightedSentence);
  }

}
