import { Pipe, PipeTransform } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

@Pipe({
  name: 'highlightWords'
})
export class HighlightWordsPipe implements PipeTransform {

  constructor(private sanitizer: DomSanitizer) {}

  transform(sentence: string, importantWords: string[]): SafeHtml {
    console.log(sentence);
    console.log(importantWords);
    if (!sentence) {
      return ' '; // or you can return an empty string as per your requirements
    }

    if (!importantWords || importantWords.length === 0) {
      return sentence; // Return the sentence as is if there are no important words
    }

    /*const regex = new RegExp(`(${importantWords.join('|')})`, 'gi');
    const highlightedSentence = sentence.replace(regex, '<span class="highlight">$1</span>');
    return this.sanitizer.bypassSecurityTrustHtml(highlightedSentence);*/
    const escapedWords = importantWords.map(word => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const regex = new RegExp(`\\b(${escapedWords.join('|')})\\b`, 'gi');
    const highlightedSentence = sentence.replace(regex, '<span class="highlight">$&</span>');
    return this.sanitizer.bypassSecurityTrustHtml(highlightedSentence);
  }

}
